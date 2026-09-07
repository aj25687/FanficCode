import re
import time
import urllib.parse
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "AO3StatsComparator/1.1 (Educational Scraping Script)"
}

# AO3 requires this cookie or it shows a "this may contain adult content"
# interstitial page instead of the real work page for Mature/Explicit works,
# which silently produces all-zero stats.
COOKIES = {"view_adult": "true"}

# Confirmed against AO3's own reference post (archiveofourown.org/admin_posts/349):
#   Categories: Gen=21, F/M=22, M/M=23, Other=24, F/F=116, Multi=2246
# The original script had F/F, F/M, and Gen scrambled.
CATEGORY_MAP = {
    "1": "23",    # M/M
    "2": "116",   # F/F
    "3": "22",    # F/M
    "4": "2246",  # Multi
    "5": "21",    # Gen
}

# Warning IDs were actually already correct in the original script, kept as-is:
# No Warnings Apply=16, Creator Chose Not To Use=14, Violence=17, MCD=18
WARNING_MAP = {
    "1": "16",
    "2": "14",
    "3": "17",
    "4": "18",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.cookies.update(COOKIES)


def get_with_retry(url, max_retries=3, backoff=10):
    """GET with basic handling for AO3 rate-limiting (429s)."""
    for attempt in range(max_retries):
        resp = SESSION.get(url)
        if resp.status_code == 429:
            wait = backoff * (attempt + 1)
            print(f"  Rate limited (429). Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        return resp
    return resp


def parse_ao3_work(work_url):
    """Fetches details and stats for a target AO3 work."""
    if "/chapters/" in work_url:
        work_url = work_url.split("/chapters/")[0]

    resp = get_with_retry(work_url)
    if resp.status_code != 200:
        raise Exception(
            f"Could not load work page. HTTP Status: {resp.status_code}"
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
        # Usually means we hit the adult-content interstitial, a login wall,
        # or the work was deleted/restricted. Fail loudly instead of
        # silently returning all-zero stats.
        raise Exception(
            "Could not find work stats on the page. The work may be "
            "restricted to logged-in users, deleted, or AO3 served an "
            "interstitial page. Check the URL in a browser."
        )

    fandoms = [a.text for a in soup.select("dd.fandom.tags ul.commas li a")]
    fandom = fandoms[0] if fandoms else ""

    def extract_stat(cls):
        node = stats_dl.find("dd", class_=cls)
        return int(node.text.replace(",", "")) if node and node.text.strip() else 0

    hits = extract_stat("hits")
    kudos = extract_stat("kudos")
    comments = extract_stat("comments")

    chapters_dd = stats_dl.find("dd", class_="chapters")
    chapters = 1
    if chapters_dd:
        match = re.search(r"(\d+)/", chapters_dd.text)
        if match:
            chapters = int(match.group(1))

    title_node = soup.find("h2", class_="title")

    return {
        "title": title_node.text.strip() if title_node else "Unknown",
        "fandom": fandom,
        "fandom_count": len(fandoms),
        "hits": hits,
        "kudos": kudos,
        "comments": comments,
        "chapters": chapters,
        "kudos_to_hits": (kudos / hits * 100) if hits > 0 else 0,
        "comments_to_hits": (comments / hits * 100) if hits > 0 else 0,
        "comments_to_kudos": (comments / kudos * 100) if kudos > 0 else 0,
    }


def build_tag_url(
    fandom, category, warning, crossover, complete_only, timeframe, page=1
):
    """Constructs an AO3 tag/search filter URL with correct AO3 parameter
    names and values (verified against AO3's own search form field list)."""
    if fandom:
        encoded_fandom = urllib.parse.quote(fandom, safe="")
        base_url = f"https://archiveofourown.org/tags/{encoded_fandom}/works?"
    else:
        base_url = "https://archiveofourown.org/works/search?"

    params = [
        ("utf8", "✓"),
        ("commit", "Sort and Filter"),
        ("page", str(page)),
    ]

    # Sorting
    if timeframe == "0":
        params.append(("work_search[sort_column]", "kudos_count"))
    else:
        params.append(("work_search[sort_column]", "revised_at"))
    params.append(("work_search[sort_direction]", "desc"))

    # Category / Warning filters
    if category in CATEGORY_MAP:
        params.append(("work_search[category_ids][]", CATEGORY_MAP[category]))

    if warning in WARNING_MAP:
        params.append(("work_search[archive_warning_ids][]", WARNING_MAP[warning]))

    # Crossover filter: AO3 expects the strings "true"/"false", not "T"/"F"
    if crossover == "1":
        params.append(("work_search[crossover]", "false"))  # exclude crossovers
    elif crossover == "2":
        params.append(("work_search[crossover]", "true"))   # crossovers only

    if complete_only == "1":
        params.append(("work_search[complete]", "1"))

    # Timeframe: AO3 has no "revised_at" text-range param. The real filter
    # is work_search[date_from], an actual date, applied against the last
    # updated date (consistent with sort_column=revised_at above).
    if timeframe in ("1", "2", "3"):
        days_map = {"1": 7, "2": 30, "3": 365}
        cutoff = datetime.today() - timedelta(days=days_map[timeframe])
        params.append(("work_search[date_from]", cutoff.strftime("%Y-%m-%d")))

    return base_url + urllib.parse.urlencode(params)


def scrape_fandom_cohort(
    fandom,
    category,
    warning,
    crossover,
    complete_only,
    timeframe,
    chapter_choice,
    target_chapters,
    max_pages=5,
):
    """Scrapes a cohort from AO3 matching category, warning, crossover, and
    chapter depth criteria."""
    fics = []

    for page in range(1, max_pages + 1):
        url = build_tag_url(
            fandom, category, warning, crossover, complete_only, timeframe, page
        )
        resp = get_with_retry(url)

        if resp.status_code != 200:
            print(f"  Page {page}: HTTP {resp.status_code}, stopping.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        work_nodes = soup.select("li.work.blurb")

        if not work_nodes:
            break

        for work in work_nodes:
            stats = work.find("dl", class_="stats")
            if not stats:
                continue

            def get_val(cls):
                el = stats.find("dd", class_=cls)
                return int(el.text.replace(",", "")) if el and el.text.strip() else 0

            hits = get_val("hits")
            kudos = get_val("kudos")
            comments = get_val("comments")

            ch_node = stats.find("dd", class_="chapters")
            chapters = 1
            if ch_node:
                m = re.search(r"(\d+)/", ch_node.text)
                if m:
                    chapters = int(m.group(1))

            # --- CHAPTER FILTERING LOGIC ---
            if chapter_choice == "1" and chapters != 1:
                continue
            elif chapter_choice == "2" and chapters != target_chapters:
                continue
            elif chapter_choice == "3" and chapters <= 1:
                continue

            # Require minimum hits to exclude newly posted works with unsettled traffic
            if hits >= 50:
                fics.append(
                    {
                        "hits": hits,
                        "kudos": kudos,
                        "comments": comments,
                        "chapters": chapters,
                        "kudos_to_hits": (kudos / hits) * 100,
                        "comments_to_hits": (comments / hits) * 100,
                        "comments_to_kudos": (comments / kudos * 100)
                        if kudos > 0
                        else 0,
                    }
                )

        time.sleep(3)

    return pd.DataFrame(fics)


def main():
    target_url = input("Enter target AO3 work URL: ").strip()

    print("\nFetching target work details...")
    target = parse_ao3_work(target_url)

    print(f"\n--- Target Fic: '{target['title']}' ---")
    print(f"Fandom: {target['fandom']}")
    print(
        f"Hits: {target['hits']} | Kudos: {target['kudos']} | "
        f"Comments: {target['comments']} | Chapters: {target['chapters']}"
    )
    print(f"Kudos/Hits Ratio: {target['kudos_to_hits']:.2f}%")
    print(f"Comments/Hits Ratio: {target['comments_to_hits']:.2f}%")

    print("\n--- Configure Filter Criteria ---")
    print("Fandom: [0] Same Fandom, [1] Search Across All Fandoms")
    fandom_choice = input("Select Fandom Scope (0-1): ").strip()
    fandom_query = target["fandom"] if fandom_choice == "0" else ""

    print(
        "\nCategory: [0] All Categories, [1] M/M, [2] F/F, [3] F/M, [4] Multi, [5] Gen"
    )
    cat_in = input("Select Category (0-5): ").strip()

    print(
        "\nArchive Warnings: [0] All Warnings, [1] No Warnings Apply, "
        "[2] Creator Chose Not To Use, [3] Violence, [4] Major Character Death"
    )
    warn_in = input("Select Warning (0-4): ").strip()

    print(
        "\nCrossovers: [0] All Works, [1] No Crossovers (Single Fandom), [2] Crossovers Only"
    )
    cross_in = input("Select Crossover Option (0-2): ").strip()

    print(
        "\nChapters: [0] All Works, [1] One-Shots Only, "
        "[2] Same Chapters as Target Fic, [3] Multi-Chapters Only"
    )
    chap_in = input("Select Chapter Filter (0-3): ").strip()

    print("\nCompletion: [0] All Works, [1] Complete Works Only")
    comp_in = input("Select Completion Status (0-1): ").strip()

    print("\nTimeframe: [0] All Time, [1] Last Week, [2] Last Month, [3] Last Year")
    time_in = input("Select Timeframe (0-3): ").strip()

    print("\nCollecting cohort comparison data from AO3...")
    df = scrape_fandom_cohort(
        fandom_query,
        cat_in,
        warn_in,
        cross_in,
        comp_in,
        time_in,
        chap_in,
        target["chapters"],
        max_pages=5,
    )

    if df.empty:
        print("No matching works found to form a comparative dataset.")
        return

    print(f"\n--- COHORT COMPARISON RESULTS (Sample size: {len(df)} works) ---")

    k_perc = (df["kudos_to_hits"] < target["kudos_to_hits"]).mean() * 100
    c_perc = (df["comments_to_hits"] < target["comments_to_hits"]).mean() * 100

    avg_k_ratio = df["kudos_to_hits"].mean()
    avg_c_ratio = df["comments_to_hits"].mean()

    print("\nKudos:Hits Ratio Comparison:")
    print(f"  Target Fic Ratio:     {target['kudos_to_hits']:.2f}%")
    print(f"  Cohort Average Ratio: {avg_k_ratio:.2f}%")
    print(
        f"  Performance:          Outperforms {k_perc:.1f}% of cohort fics "
        f"(Top {100 - k_perc:.1f}%)"
    )

    print("\nComments:Hits Ratio Comparison:")
    print(f"  Target Fic Ratio:     {target['comments_to_hits']:.2f}%")
    print(f"  Cohort Average Ratio: {avg_c_ratio:.2f}%")
    print(
        f"  Performance:          Outperforms {c_perc:.1f}% of cohort fics "
        f"(Top {100 - c_perc:.1f}%)"
    )


if __name__ == "__main__":
    main()
