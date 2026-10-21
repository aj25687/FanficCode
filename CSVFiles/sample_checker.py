
#!/usr/bin/env python3
"""
ao3_sample_checker.py
=====================
Compares your hand-coding sample CSV against known AO3 fandom population
numbers to verify the sample is proportionally representative.

Checks: ratings, warnings, categories, top 10 characters, top 10 relationships.

For multi-tag fields (warnings, categories, characters, relationships),
the comparison uses ONLY tags that match the known top-N list so the
percentages are apples-to-apples with the population numbers.

Usage:
    python3 ao3_sample_checker.py
    python3 ao3_sample_checker.py --csv path/to/ao3_testing_sample.csv
"""

import argparse
import csv
import os
import sys

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Check AO3 sample representativeness.")
parser.add_argument("--csv", default="ao3_testing_sample.csv", help="Sample CSV to check.")
parser.add_argument("--sep", default=" | ", help="Tag separator in CSV (default: ' | ').")
ARGS = parser.parse_args()

CSV_PATH = os.path.abspath(ARGS.csv)
TAG_SEP = ARGS.sep


# ══════════════════════════════════════════════════════════════
# POPULATION DATA  (from AO3 filter sidebar counts)
# ══════════════════════════════════════════════════════════════

FANDOMS = {
    "Harry Potter": {
        "total": 718412,
        "fandom_keywords": ["harry potter", "j. k. rowling", "j.k. rowling"],
        "ratings": {
            "Teen And Up Audiences": 189685,
            "General Audiences": 176597,
            "Explicit": 133153,
            "Mature": 120658,
            "Not Rated": 98319,
        },
        "warnings": {
            "No Archive Warnings Apply": 347727,
            "Creator Chose Not To Use Archive Warnings": 275175,
            "Graphic Depictions Of Violence": 65340,
            "Major Character Death": 50966,
            "Underage Sex": 30442,
            "Rape/Non-Con": 21679,
        },
        "categories": {
            "M/M": 302142,
            "F/M": 269110,
            "Gen": 110632,
            "F/F": 62690,
            "Multi": 48043,
            "Other": 18513,
        },
        "characters": {
            "Harry Potter": 293145,
            "Hermione Granger": 206057,
            "Draco Malfoy": 189729,
            "Sirius Black": 145397,
            "Ron Weasley": 131411,
            "Remus Lupin": 128165,
            "Severus Snape": 117013,
            "James Potter": 109199,
            "Lily Evans Potter": 83830,
            "Ginny Weasley": 79230,
        },
        "relationships": {
            "Draco Malfoy/Harry Potter": 89069,
            "Sirius Black/Remus Lupin": 73426,
            "Hermione Granger/Draco Malfoy": 49111,
            "James Potter/Lily Evans Potter": 48996,
            "Hermione Granger/Ron Weasley": 30201,
            "Regulus Black/James Potter": 28627,
            "Harry Potter/Ginny Weasley": 25355,
            "Harry Potter/Severus Snape": 22132,
            "Marlene McKinnon/Dorcas Meadowes": 12762,
            "Hermione Granger/Harry Potter": 12633,
        },
    },
    "Arcane": {
        "total": 66102,
        "fandom_keywords": ["arcane", "league of legends"],
        "ratings": {
            "Explicit": 21427,
            "Teen And Up Audiences": 14870,
            "Mature": 12456,
            "General Audiences": 10839,
            "Not Rated": 6510,
        },
        "warnings": {
            "No Archive Warnings Apply": 34360,
            "Creator Chose Not To Use Archive Warnings": 23255,
            "Graphic Depictions Of Violence": 7461,
            "Major Character Death": 3809,
            "Rape/Non-Con": 1671,
            "Underage Sex": 1081,
        },
        "categories": {
            "F/F": 27553,
            "M/M": 23977,
            "F/M": 13050,
            "Gen": 6747,
            "Multi": 3066,
            "Other": 2112,
        },
        "characters": {
            "Vi (League of Legends)": 28506,
            "Jayce (League of Legends)": 28102,
            "Viktor (League of Legends)": 26958,
            "Caitlyn (League of Legends)": 26545,
            "Jinx (League of Legends)": 23312,
            "Ekko (League of Legends)": 12880,
            "Silco (Arcane: League of Legends)": 11208,
            "Mel Medarda": 10239,
            "Sevika (Arcane: League of Legends)": 8786,
            "Vander (League of Legends)": 8411,
        },
        "relationships": {
            "Jayce/Viktor (League of Legends)": 21523,
            "Caitlyn/Vi (League of Legends)": 21422,
            "Ekko/Jinx (League of Legends)": 6878,
            "Jayce & Viktor (League of Legends)": 4781,
            "Jinx & Vi (League of Legends)": 3906,
            "Caitlyn & Vi (League of Legends)": 3577,
            "Silco (Arcane: League of Legends)/Vander (League of Legends)": 2251,
            "Jayce (League of Legends)/Mel Medarda": 2186,
            "Jinx (League of Legends) & Silco (Arcane: League of Legends)": 1999,
            'Luxanna "Lux" Crownguard/Jinx': 1807,
        },
    },
    "Avatar": {
        "total": 57034,
        "fandom_keywords": ["avatar", "last airbender", "atla"],
        "ratings": {
            "Teen And Up Audiences": 20242,
            "General Audiences": 18978,
            "Mature": 6802,
            "Explicit": 6219,
            "Not Rated": 4793,
        },
        "warnings": {
            "No Archive Warnings Apply": 33323,
            "Creator Chose Not To Use Archive Warnings": 16761,
            "Graphic Depictions Of Violence": 5419,
            "Major Character Death": 3362,
            "Underage Sex": 1727,
            "Rape/Non-Con": 1316,
        },
        "categories": {
            "F/M": 23982,
            "Gen": 16776,
            "M/M": 12764,
            "F/F": 7732,
            "Multi": 3534,
            "Other": 1418,
        },
        "characters": {
            "Zuko (Avatar)": 38180,
            "Katara (Avatar)": 26441,
            "Sokka (Avatar)": 24348,
            "Aang (Avatar)": 20316,
            "Toph Beifong": 15452,
            "Azula (Avatar)": 15154,
            "Iroh (Avatar)": 12666,
            "Suki (Avatar)": 10342,
            "Mai (Avatar)": 8881,
            "Ty Lee (Avatar)": 7738,
        },
        "relationships": {
            "Katara/Zuko (Avatar)": 9435,
            "Sokka/Zuko (Avatar)": 8904,
            "Aang/Katara (Avatar)": 6539,
            "Sokka/Suki (Avatar)": 4051,
            "Mai/Zuko (Avatar)": 3516,
            "Iroh & Zuko (Avatar)": 3402,
            "Azula & Zuko (Avatar)": 3228,
            "The Gaang & Zuko (Avatar)": 2556,
            "Katara & Zuko (Avatar)": 2214,
            "Sokka & Zuko (Avatar)": 2086,
        },
    },
}


# ══════════════════════════════════════════════════════════════
# CSV READING
# ══════════════════════════════════════════════════════════════

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        rows = []
        for raw in reader:
            cleaned = {}
            for k, v in raw.items():
                key = k.strip() if k else k
                val = v.strip() if isinstance(v, str) else ""
                if val.lower() in ("nan", "none", "null"):
                    val = ""
                cleaned[key] = val
            rows.append(cleaned)
    return rows


def detect_fandom(row):
    fandom_text = row.get("fandom", "").lower()
    for name, info in FANDOMS.items():
        for keyword in info["fandom_keywords"]:
            if keyword in fandom_text:
                return name
    return "Unknown"


def split_tags(value):
    if not value:
        return []
    return [t.strip() for t in value.split(TAG_SEP) if t.strip()]


# ══════════════════════════════════════════════════════════════
# MATCHING & COUNTING
# ══════════════════════════════════════════════════════════════

def fuzzy_match_tag(tag, known_tags):
    tag_lower = tag.lower().strip()
    for known in known_tags:
        if tag_lower == known.lower().strip():
            return known
    for known in known_tags:
        known_lower = known.lower().strip()
        if tag_lower in known_lower or known_lower in tag_lower:
            return known
    return None


def count_matched_tags(rows, field, pop_tags):
    """Count sample tags matching population tags.
    Returns:
      counts       = {pop_tag: n_matches}
      matched_total = number of tag occurrences that matched ANY known tag
                      (this is the correct denominator for apples-to-apples comparison)
      raw_total     = total tag occurrences (for the coverage stat)
    """
    counts = {tag: 0 for tag in pop_tags}
    matched_total = 0
    raw_total = 0
    for row in rows:
        tags = split_tags(row.get(field, ""))
        if not tags:
            raw = row.get(field, "").strip()
            if raw:
                tags = [raw]
        for tag in tags:
            raw_total += 1
            match = fuzzy_match_tag(tag, pop_tags)
            if match:
                counts[match] += 1
                matched_total += 1
    return counts, matched_total, raw_total


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: {CSV_PATH} not found.")

    rows = read_csv(CSV_PATH)
    print(f"Loaded {len(rows)} works from {os.path.basename(CSV_PATH)}\n")

    # Group rows by fandom
    fandom_rows = {}
    unknown = []
    for row in rows:
        fandom = detect_fandom(row)
        if fandom == "Unknown":
            unknown.append(row)
        else:
            fandom_rows.setdefault(fandom, []).append(row)

    print("=" * 70)
    print("FANDOM BREAKDOWN IN SAMPLE")
    print("=" * 70)
    for name in FANDOMS:
        n = len(fandom_rows.get(name, []))
        pop = FANDOMS[name]["total"]
        print(f"  {name:>15}: {n:>4} sampled  /  {pop:>7,} population")
    if unknown:
        print(f"  {'Unknown':>15}: {len(unknown):>4} sampled  (no population data)")
    print(f"  {'TOTAL':>15}: {len(rows):>4} sampled")

    # ── Check each fandom ────────────────────────────────────
    for fandom_name, fandom_info in FANDOMS.items():
        f_rows = fandom_rows.get(fandom_name, [])
        if not f_rows:
            print(f"\n{'=' * 70}")
            print(f"SKIPPING {fandom_name}: no works in sample")
            continue

        n = len(f_rows)
        print(f"\n{'=' * 70}")
        print(f"{fandom_name.upper()}  ({n} sampled  /  {fandom_info['total']:,} population)")
        print(f"{'=' * 70}")

        dimensions = [
            ("RATINGS",       "rating",        fandom_info["ratings"],       True),
            ("WARNINGS",      "warnings",      fandom_info["warnings"],      False),
            ("CATEGORIES",    "category",      fandom_info["categories"],    False),
            ("CHARACTERS",    "characters",    fandom_info["characters"],    False),
            ("RELATIONSHIPS", "relationships", fandom_info["relationships"], False),
        ]

        for dim_name, field, pop_counts, is_single_value in dimensions:
            print(f"\n  {dim_name}")
            print(f"  {'-' * 66}")

            sample_counts, matched_total, raw_total = count_matched_tags(
                f_rows, field, pop_counts.keys()
            )

            # ── Denominator logic ────────────────────────────
            # For single value fields (rating): each work has exactly 1 → use row count.
            # For multi tag fields: use ONLY the matched tag count so
            #   percentages are comparable to the population (which also
            #   only counts the listed tags, not "everything else").
            if is_single_value:
                denominator = n
                coverage_note = ""
            else:
                denominator = matched_total if matched_total > 0 else 1
                coverage_pct = matched_total / raw_total * 100 if raw_total > 0 else 0
                coverage_note = (
                    f"  (top-N tags cover {matched_total} of {raw_total} "
                    f"tag slots = {coverage_pct:.0f}% coverage)"
                )

            pop_total = sum(pop_counts.values())

            header = (f"  {'Tag':<50} {'Pop%':>6} {'Samp%':>6} "
                      f"{'Diff':>7}  {'':>3}  {'Count':>8}")
            print(header)

            max_diff = 0
            for tag, pop_count in pop_counts.items():
                pop_pct = pop_count / pop_total * 100 if pop_total > 0 else 0
                samp_count = sample_counts.get(tag, 0)
                samp_pct = samp_count / denominator * 100 if denominator > 0 else 0
                diff = samp_pct - pop_pct
                abs_diff = abs(diff)
                max_diff = max(max_diff, abs_diff)

                if abs_diff > 10:
                    flag = "!!!"
                elif abs_diff > 5:
                    flag = " ! "
                elif abs_diff > 3:
                    flag = " ~ "
                else:
                    flag = " ok"

                print(f"  {tag:<50} {pop_pct:5.1f}% {samp_pct:5.1f}% "
                      f"{diff:+6.1f}%  {flag}  {samp_count:>4}/{denominator}")

            # Dimension verdict
            if max_diff <= 3:
                verdict = "GOOD"
            elif max_diff <= 5:
                verdict = "ACCEPTABLE"
            elif max_diff <= 10:
                verdict = "WATCH"
            else:
                verdict = "PROBLEM"
            print(f"  >> {dim_name} verdict: {verdict} (worst deviation: {max_diff:.1f} pp)")
            if coverage_note:
                print(f"  {coverage_note}")

    # ── Legend ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("LEGEND")
    print(f"{'=' * 70}")
    print("   ok  = within 3 pp  (representative)")
    print("    ~  = 3 to 5 pp    (acceptable, monitor)")
    print("    !  = 5 to 10 pp   (notable deviation)")
    print("   !!! = >10 pp       (sample may not be representative)")
    print()
    print("HOW IT WORKS:")
    print("  For multi tag fields (warnings, categories, characters,")
    print("  relationships), percentages are calculated ONLY among tags")
    print("  that matched the known top N list. This makes the comparison")
    print("  apples to apples with the AO3 population numbers.")
    print()
    print("  The 'coverage' line shows what fraction of your sample's tags")
    print("  fell within the top N. Low coverage is fine; it just means")
    print("  your sample has lots of less common tags (expected).")
    print()
    print("  For ratings, each work has exactly one, so the denominator")
    print("  is simply the number of works.")


if __name__ == "__main__":
    main()

