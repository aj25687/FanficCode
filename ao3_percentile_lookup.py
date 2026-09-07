"""
Offline lookup against a saved AO3 distribution (ao3_distributions.json).

Run ao3_stats_comparator.py once per fandom+filter combo to build up the
distribution file. After that, use this script as many times as you like
to check where any fic falls on that distribution -- no AO3 requests at all.

Usage:
    python3 ao3_percentile_lookup.py
"""

import bisect
import json
import os

DIST_FILE = "ao3_distributions.json"


def percentile_rank(value, sorted_values):
    """% of sorted_values strictly less than value."""
    if not sorted_values:
        return None
    idx = bisect.bisect_left(sorted_values, value)
    return idx / len(sorted_values) * 100


def main():
    if not os.path.exists(DIST_FILE):
        print(f"No {DIST_FILE} found yet. Run ao3_stats_comparator.py at least once first.")
        return

    with open(DIST_FILE, "r", encoding="utf-8") as f:
        all_dist = json.load(f)

    if not all_dist:
        print(f"{DIST_FILE} exists but has no saved distributions yet.")
        return

    keys = list(all_dist.keys())
    print("Saved distributions:")
    for i, k in enumerate(keys):
        entry = all_dist[k]
        print(f"  [{i}] {k}  (sample size: {entry['sample_size']}, updated {entry['last_updated']})")

    choice = input("\nWhich distribution do you want to check against (number)? ").strip()
    try:
        entry = all_dist[keys[int(choice)]]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    hits = int(input("Fic's hits: ").strip())
    kudos = int(input("Fic's kudos: ").strip())
    comments = int(input("Fic's comments: ").strip())

    kudos_to_hits = (kudos / hits * 100) if hits else 0
    comments_to_hits = (comments / hits * 100) if hits else 0
    comments_to_kudos = (comments / kudos * 100) if kudos else 0

    print(f"\n--- Ratios ---")
    print(f"Kudos:Hits     {kudos_to_hits:.2f}%")
    print(f"Comments:Hits  {comments_to_hits:.2f}%")
    print(f"Comments:Kudos {comments_to_kudos:.2f}%")

    print(f"\n--- Percentile vs. saved distribution (n={entry['sample_size']}) ---")
    for label, ratio_key, value in [
        ("Kudos:Hits", "kudos_to_hits", kudos_to_hits),
        ("Comments:Hits", "comments_to_hits", comments_to_hits),
        ("Comments:Kudos", "comments_to_kudos", comments_to_kudos),
    ]:
        values = entry["ratios"][ratio_key]["values"]
        pct = percentile_rank(value, values)
        mean = entry["ratios"][ratio_key]["mean"]
        print(
            f"{label:15s} outperforms {pct:5.1f}% of cohort "
            f"(cohort mean: {mean:.2f}%, this fic: {value:.2f}%)"
        )


if __name__ == "__main__":
    main()
