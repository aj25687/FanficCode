import re
import time
import urllib.parse
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import pandas as pd
from curl_cffi import requests

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

SESSION = requests.Session(impersonate="chrome120")
SESSION.headers.update(HEADERS)
SESSION.cookies.update(COOKIES)


def parse_cookie_string(cookie_str):
    """Parses a raw 'Cookie:' header value (as copied from browser dev
    tools, e.g. "_otwarchive_session=abc123; remember_user_token=xyz789")
    into a dict suitable for requests.Session.cookies.update()."""
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        cookies[key.strip()] = value.strip()
    return cookies


def apply_session_cookie(cookie_str):
    """Adds a user-supplied logged-in AO3 session cookie to the shared
    SESSION, so subsequent requests are authenticated and can see works
    marked 'restricted to registered users' that anonymous requests can't.
    """
    if not cookie_str:
        return
    parsed = parse_cookie_string(cookie_str)
    if parsed:
        SESSION.cookies.update(parsed)
        print(f"  Applied session cookie ({len(parsed)} field(s)). Requests will be sent as a logged-in user.")
    else:
        print("  Could not parse that cookie string — continuing without it (anonymous requests).")


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
    """Constructs an AO3 work-search filter URL with correct AO3 parameter
    names and values (verified against AO3's own search form field list).

    Always uses /works/search rather than /tags/<name>/works. AO3's tag-path
    URLs require special escaping for characters like periods, slashes, and
    ampersands in tag names (e.g. "." becomes "*d*"), which plain percent-
    encoding doesn't replicate and which causes 404s for very common fandom
    names (e.g. "Harry Potter - J. K. Rowling"). Passing the fandom name as
    the work_search[fandom_names] query parameter avoids that entirely,
    since it's just a text field with normal URL-encoding.
    """
    base_url = "https://archiveofourown.org/works/search?"

    params = [
        ("utf8", "✓"),
        ("commit", "Sort and Filter"),
        ("page", str(page)),
    ]

    if fandom:
        # Wrapped in quotes for an exact-phrase match against the fandom
        # tag rather than a loose keyword search, which is closer to what
        # AO3's own tag-page browsing does.
        params.append(("work_search[fandom_names]", f'"{fandom}"'))

    # Sorting: neutral (last-updated). Since scrape_fandom_cohort() now walks
    # every result page rather than taking a sample, the sort order here has
    # no effect on which works end up in the cohort — it's only relevant to
    # the order pages are returned in.
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


def get_result_page_count(fandom, category, warning, crossover, complete_only, timeframe):
    """Fetches page 1 of the filtered search and reads AO3's own pagination
    controls to find the total number of result pages, so scrape_fandom_cohort
    knows how many pages it needs to walk to cover every matching work."""
    url = build_tag_url(fandom, category, warning, crossover, complete_only, timeframe, page=1)
    resp = get_with_retry(url)

    if resp.status_code != 200:
        return 0, resp

    soup = BeautifulSoup(resp.text, "html.parser")
    if not soup.select("li.work.blurb"):
        return 0, resp

    page_links = soup.select("ol.pagination li a")
    page_numbers = [int(a.text) for a in page_links if a.text.strip().isdigit()]
    total_pages = max(page_numbers) if page_numbers else 1
    return total_pages, resp


def scrape_fandom_cohort(
    fandom,
    category,
    warning,
    crossover,
    complete_only,
    timeframe,
    chapter_choice,
    target_chapters,
    max_pages=None,
):
    """Scrapes EVERY work on AO3 matching the given category, warning,
    crossover, and chapter-depth criteria (not a sample).

    Walks every result page of the filtered search sequentially. Sort order
    doesn't matter for correctness here since we're covering the whole
    result set — sorting only mattered when this used to take a partial
    sample and needed to avoid a popularity/sort bias.

    max_pages, if given, caps how many pages are fetched (mainly useful for
    testing on a smaller slice); leave it as None to fetch everything.
    """
    fics = []

    total_pages, first_page_resp = get_result_page_count(
        fandom, category, warning, crossover, complete_only, timeframe
    )

    if total_pages == 0:
        return pd.DataFrame(fics)

    pages_to_fetch = list(range(1, total_pages + 1))
    if max_pages is not None:
        pages_to_fetch = pages_to_fetch[:max_pages]

    est_seconds = len(pages_to_fetch) * 3
    print(
        f"  Found {total_pages} result page(s) (~{total_pages * 20} works). "
        f"Fetching {len(pages_to_fetch)} page(s), roughly {est_seconds // 60}m "
        f"{est_seconds % 60}s minimum at AO3's rate limit..."
    )

    for i, page in enumerate(pages_to_fetch, start=1):
        # Reuse the already-fetched page 1 response instead of refetching it.
        if page == 1:
            resp = first_page_resp
        else:
            url = build_tag_url(
                fandom, category, warning, crossover, complete_only, timeframe, page
            )
            resp = get_with_retry(url)

        if resp.status_code != 200:
            print(f"  Page {page}: HTTP {resp.status_code}, skipping.")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        work_nodes = soup.select("li.work.blurb")

        if not work_nodes:
            continue

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

        if i % 10 == 0 or i == len(pages_to_fetch):
            print(f"  ...page {i}/{len(pages_to_fetch)} done, {len(fics)} qualifying works so far")

        time.sleep(5)

    return pd.DataFrame(fics)


def main():
    print("Optional: paste a logged-in AO3 session cookie to include works")
    print("restricted to registered users (copy the 'Cookie' request header")
    print("from your browser's dev tools while logged into AO3).")
    print("Leave blank to continue anonymously (restricted works excluded).")
    cookie_in = input("AO3 session cookie (optional): ").strip()
    apply_session_cookie(cookie_in)

    target_url = input("\nEnter target AO3 work URL: ").strip()

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

    print(
        "\nCollecting cohort comparison data from AO3 "
        "(scanning every matching work — this can take a while)..."
    )
    df = scrape_fandom_cohort(
        fandom_query,
        cat_in,
        warn_in,
        cross_in,
        comp_in,
        time_in,
        chap_in,
        target["chapters"],
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