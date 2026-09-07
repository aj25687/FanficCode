import math
import re
import time
import urllib.parse
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "AO3StatsComparator/1.0 (Educational Scraping Script)"
}


def parse_ao3_work(work_url):
    """Fetches details and stats for a single target AO3 work."""
    if "/chapters/" in work_url:
        work_url = work_url.split("/chapters/")[0]

    resp = requests.get(work_url, headers=HEADERS)
    if resp.status_code != 200:
        raise Exception(
            f"Could not load work page. HTTP Status: {resp.status_code}"
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    fandoms = [
        a.text for a in soup.select("dd.fandom.tags ul.commas li a")
    ]
    fandom = fandoms[0] if fandoms else ""

    stats_dl = soup.find("dl", class_="stats")

    def extract_stat(cls):
        node = stats_dl.find("dd", class_=cls) if stats_dl else None
        return int(node.text.replace(",", "")) if node else 0

    hits = extract_stat("hits")
    kudos = extract_stat("kudos")
    comments = extract_stat("comments")

    chapters_dd = stats_dl.find("dd", class_="chapters") if stats_dl else None
    chapters = 1
    if chapters_dd:
        match = re.search(r"(\d+)/", chapters_dd.text)
        if match:
            chapters = int(match.group(1))

    return {
        "title": soup.find("h2", class_="title").text.strip()
        if soup.find("h2", class_="title")
        else "Unknown",
        "fandom": fandom,
        "hits": hits,
        "kudos": kudos,
        "comments": comments,
        "chapters": chapters,
        "kudos_to_hits": (kudos / hits * 100) if hits > 0 else 0,
        "comments_to_hits": (comments / hits * 100) if hits > 0 else 0,
        "comments_to_kudos": (comments / kudos * 100) if kudos > 0 else 0,
    }


def build_search_url(
    fandom, category, warning, complete_only, timeframe, page=1
):
    """Constructs the search URL for AO3 based on user criteria."""
    base_url = "https://archiveofourown.org/works/search?"

    # Category Mapping (Internal AO3 Database IDs)
    cat_map = {
        "1": "23",  # M/M
        "2": "22",  # F/F
        "3": "21",  # F/M
        "4": "2246",  # Multi
        "5": "24",  # Gen
    }

    # Archive Warning Mapping (Internal AO3 Database IDs)
    warn_map = {
        "1": "16",  # No Archive Warnings Apply
        "2": "14",  # Creator Chose Not To Use
        "3": "17",  # Graphic Depictions Of Violence
        "4": "18",  # Major Character Death
    }

    params = [
        ("commit", "Search"),
        ("page", str(page)),
        ("work_search[fandom_names]", fandom if fandom else ""),
        ("work_search[sort_column]", "revised_at"),
        ("work_search[sort_direction]", "desc"),
    ]

    # Only filter if the user did NOT select 0 ("All")
    if category in cat_map:
        params.append(("work_search[category_ids][]", cat_map[category]))

    if warning in warn_map:
        params.append(
            ("work_search[archive_warning_ids][]", warn_map[warning])
        )

    if complete_only == "1":
        params.append(("work_search[complete]", "1"))

    # Timeframe handling
    if timeframe == "1":
        params.append(("work_search[revised_at]", "< 1 week"))
    elif timeframe == "2":
        params.append(("work_search[revised_at]", "< 1 month"))
    elif timeframe == "3":
        params.append(("work_search[revised_at]", "< 1 year"))

    return base_url + urllib.parse.urlencode(params)


def scrape_fandom_cohort(
    fandom, category, warning, complete_only, timeframe, max_pages=3
):
    """Collects baseline data from matching AO3 search pages."""
    fics = []

    for page in range(1, max_pages + 1):
        url = build_search_url(
            fandom, category, warning, complete_only, timeframe, page
        )
        resp = requests.get(url, headers=HEADERS)

        if resp.status_code != 200:
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
                return int(el.text.replace(",", "")) if el else 0

            hits = get_val("hits")
            kudos = get_val("kudos")
            comments = get_val("comments")

            ch_node = stats.find("dd", class_="chapters")
            chapters = 1
            if ch_node:
                m = re.search(r"(\d+)/", ch_node.text)
                if m:
                    chapters = int(m.group(1))

            if hits > 0:
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

        # 3-second delay to comply with AO3 rate limits
        time.sleep(3)

    return pd.DataFrame(fics)


def main():
    target_url = input("Enter target AO3 work URL: ").strip()

    print("\nFetching target work details...")
    target = parse_ao3_work(target_url)

    print(f"\n--- Target Fic: '{target['title']}' ---")
    print(f"Fandom: {target['fandom']}")
    print(
        f"Hits: {target['hits']} | Kudos: {target['kudos']} | Comments: {target['comments']} | Chapters: {target['chapters']}"
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
        "\nArchive Warnings: [0] All Warnings, [1] No Warnings Apply, [2] Creator Chose Not To Use, [3] Violence, [4] Major Character Death"
    )
    warn_in = input("Select Warning (0-4): ").strip()

    print("\nCompletion: [0] All Works, [1] Complete Works Only")
    comp_in = input("Select Completion Status (0-1): ").strip()

    print(
        "\nTimeframe: [0] All Time, [1] Last Week, [2] Last Month, [3] Last Year"
    )
    time_in = input("Select Timeframe (0-3): ").strip()

    print("\nCollecting cohort comparison data from AO3...")
    df = scrape_fandom_cohort(
        fandom_query, cat_in, warn_in, comp_in, time_in, max_pages=3
    )

    if df.empty:
        print("No matching works found to form a comparative dataset.")
        return

    print(
        f"\n--- COHORT COMPARISON RESULTS (Sample size: {len(df)} works) ---"
    )

    # Calculate Percentiles
    k_perc = (
        (df["kudos_to_hits"] < target["kudos_to_hits"]).mean() * 100
    )
    c_perc = (
        (df["comments_to_hits"] < target["comments_to_hits"]).mean()
        * 100
    )

    avg_k_ratio = df["kudos_to_hits"].mean()
    avg_c_ratio = df["comments_to_hits"].mean()

    print(f"\nKudos:Hits Ratio Comparison:")
    print(f"  Target Fic Ratio:     {target['kudos_to_hits']:.2f}%")
    print(f"  Cohort Average Ratio: {avg_k_ratio:.2f}%")
    print(
        f"  Performance:          Outperforms {k_perc:.1f}% of cohort fics (Top {100 - k_perc:.1f}%)"
    )

    print(f"\nComments:Hits Ratio Comparison:")
    print(f"  Target Fic Ratio:     {target['comments_to_hits']:.2f}%")
    print(f"  Cohort Average Ratio: {avg_c_ratio:.2f}%")
    print(
        f"  Performance:          Outperforms {c_perc:.1f}% of cohort fics (Top {100 - c_perc:.1f}%)"
    )


if __name__ == "__main__":
    main()
