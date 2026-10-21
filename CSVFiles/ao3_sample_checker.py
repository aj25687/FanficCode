
#!/usr/bin/env python3
"""
ao3_sample_checker.py
=====================
Compares your hand-coding sample CSV against known AO3 fandom population
numbers to verify the sample is proportionally representative.

Outputs:
  - checker_summary.txt  (human readable report, overwritten each run)
  - checker_results.json (structured data for ao3_sample_fixer.py)

Usage:
    python3 ao3_sample_checker.py
    python3 ao3_sample_checker.py --csv path/to/ao3_testing_sample.csv
"""

import argparse
import csv
import json
import os
import sys

parser = argparse.ArgumentParser(description="Check AO3 sample representativeness.")
parser.add_argument("--csv", default="ao3_testing_sample.csv")
parser.add_argument("--sep", default=" | ")
parser.add_argument("--summary", default="checker_summary.txt")
parser.add_argument("--json-out", default="checker_results.json")
ARGS = parser.parse_args()

CSV_PATH = os.path.abspath(ARGS.csv)
TAG_SEP = ARGS.sep


# ── Tee: print to stdout AND file simultaneously ────────────
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for s in self.streams:
            s.write(text)
    def flush(self):
        for s in self.streams:
            s.flush()


# ══════════════════════════════════════════════════════════════
# POPULATION DATA (from AO3 filter sidebar counts)
# ══════════════════════════════════════════════════════════════

FANDOMS = {
    "Harry Potter": {
        "total": 718412,
        "fandom_keywords": ["harry potter", "j. k. rowling", "j.k. rowling"],
        "ratings": {"Teen And Up Audiences": 189685, "General Audiences": 176597, "Explicit": 133153, "Mature": 120658, "Not Rated": 98319},
        "warnings": {"No Archive Warnings Apply": 347727, "Creator Chose Not To Use Archive Warnings": 275175, "Graphic Depictions Of Violence": 65340, "Major Character Death": 50966, "Underage Sex": 30442, "Rape/Non-Con": 21679},
        "categories": {"M/M": 302142, "F/M": 269110, "Gen": 110632, "F/F": 62690, "Multi": 48043, "Other": 18513},
        "characters": {"Harry Potter": 293145, "Hermione Granger": 206057, "Draco Malfoy": 189729, "Sirius Black": 145397, "Ron Weasley": 131411, "Remus Lupin": 128165, "Severus Snape": 117013, "James Potter": 109199, "Lily Evans Potter": 83830, "Ginny Weasley": 79230},
        "relationships": {"Draco Malfoy/Harry Potter": 89069, "Sirius Black/Remus Lupin": 73426, "Hermione Granger/Draco Malfoy": 49111, "James Potter/Lily Evans Potter": 48996, "Hermione Granger/Ron Weasley": 30201, "Regulus Black/James Potter": 28627, "Harry Potter/Ginny Weasley": 25355, "Harry Potter/Severus Snape": 22132, "Marlene McKinnon/Dorcas Meadowes": 12762, "Hermione Granger/Harry Potter": 12633},
    },
    "Arcane": {
        "total": 66102,
        "fandom_keywords": ["arcane", "league of legends"],
        "ratings": {"Explicit": 21427, "Teen And Up Audiences": 14870, "Mature": 12456, "General Audiences": 10839, "Not Rated": 6510},
        "warnings": {"No Archive Warnings Apply": 34360, "Creator Chose Not To Use Archive Warnings": 23255, "Graphic Depictions Of Violence": 7461, "Major Character Death": 3809, "Rape/Non-Con": 1671, "Underage Sex": 1081},
        "categories": {"F/F": 27553, "M/M": 23977, "F/M": 13050, "Gen": 6747, "Multi": 3066, "Other": 2112},
        "characters": {"Vi (League of Legends)": 28506, "Jayce (League of Legends)": 28102, "Viktor (League of Legends)": 26958, "Caitlyn (League of Legends)": 26545, "Jinx (League of Legends)": 23312, "Ekko (League of Legends)": 12880, "Silco (Arcane: League of Legends)": 11208, "Mel Medarda": 10239, "Sevika (Arcane: League of Legends)": 8786, "Vander (League of Legends)": 8411},
        "relationships": {"Jayce/Viktor (League of Legends)": 21523, "Caitlyn/Vi (League of Legends)": 21422, "Ekko/Jinx (League of Legends)": 6878, "Jayce & Viktor (League of Legends)": 4781, "Jinx & Vi (League of Legends)": 3906, "Caitlyn & Vi (League of Legends)": 3577, "Silco (Arcane: League of Legends)/Vander (League of Legends)": 2251, "Jayce (League of Legends)/Mel Medarda": 2186, "Jinx (League of Legends) & Silco (Arcane: League of Legends)": 1999, 'Luxanna "Lux" Crownguard/Jinx': 1807},
    },
    "Avatar": {
        "total": 57034,
        "fandom_keywords": ["avatar", "last airbender", "atla"],
        "ratings": {"Teen And Up Audiences": 20242, "General Audiences": 18978, "Mature": 6802, "Explicit": 6219, "Not Rated": 4793},
        "warnings": {"No Archive Warnings Apply": 33323, "Creator Chose Not To Use Archive Warnings": 16761, "Graphic Depictions Of Violence": 5419, "Major Character Death": 3362, "Underage Sex": 1727, "Rape/Non-Con": 1316},
        "categories": {"F/M": 23982, "Gen": 16776, "M/M": 12764, "F/F": 7732, "Multi": 3534, "Other": 1418},
        "characters": {"Zuko (Avatar)": 38180, "Katara (Avatar)": 26441, "Sokka (Avatar)": 24348, "Aang (Avatar)": 20316, "Toph Beifong": 15452, "Azula (Avatar)": 15154, "Iroh (Avatar)": 12666, "Suki (Avatar)": 10342, "Mai (Avatar)": 8881, "Ty Lee (Avatar)": 7738},
        "relationships": {"Katara/Zuko (Avatar)": 9435, "Sokka/Zuko (Avatar)": 8904, "Aang/Katara (Avatar)": 6539, "Sokka/Suki (Avatar)": 4051, "Mai/Zuko (Avatar)": 3516, "Iroh & Zuko (Avatar)": 3402, "Azula & Zuko (Avatar)": 3228, "The Gaang & Zuko (Avatar)": 2556, "Katara & Zuko (Avatar)": 2214, "Sokka & Zuko (Avatar)": 2086},
    },
}


# ══════════════════════════════════════════════════════════════
# CSV & TAG HELPERS
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
        for kw in info["fandom_keywords"]:
            if kw in fandom_text:
                return name
    return "Unknown"


def split_tags(value):
    if not value:
        return []
    return [t.strip() for t in value.split(TAG_SEP) if t.strip()]


def fuzzy_match_tag(tag, known_tags):
    tag_lower = tag.lower().strip()
    for known in known_tags:
        if tag_lower == known.lower().strip():
            return known
    for known in known_tags:
        if tag_lower in known.lower().strip() or known.lower().strip() in tag_lower:
            return known
    return None


def count_matched_tags(rows, field, pop_tags):
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
# CORE CHECK LOGIC (importable)
# ══════════════════════════════════════════════════════════════

def run_check(csv_path, tag_sep=" | "):
    """Run the full representativeness check.
    Prints results to stdout (and whatever stdout is tee'd to).
    Returns structured results dict for programmatic use."""
    global TAG_SEP
    TAG_SEP = tag_sep

    rows = read_csv(csv_path)
    print(f"Loaded {len(rows)} works from {os.path.basename(csv_path)}\n")

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

    results = {}

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

        fandom_result = {
            "n_works": n,
            "pop_total": fandom_info["total"],
            "dimensions": {},
        }

        dimensions = [
            ("ratings",       "rating",        fandom_info["ratings"],       True),
            ("warnings",      "warnings",      fandom_info["warnings"],      False),
            ("categories",    "category",      fandom_info["categories"],    False),
            ("characters",    "characters",    fandom_info["characters"],    False),
            ("relationships", "relationships", fandom_info["relationships"], False),
        ]

        for dim_name, field, pop_counts, is_single in dimensions:
            print(f"\n  {dim_name.upper()}")
            print(f"  {'-' * 66}")

            sample_counts, matched_total, raw_total = count_matched_tags(
                f_rows, field, pop_counts.keys()
            )

            if is_single:
                denominator = n
            else:
                denominator = matched_total if matched_total > 0 else 1

            pop_total = sum(pop_counts.values())

            header = (f"  {'Tag':<50} {'Pop%':>6} {'Samp%':>6} "
                      f"{'Diff':>7}  {'':>3}  {'Count':>8}")
            print(header)

            dim_result = {
                "field": field,
                "is_single_value": is_single,
                "worst_diff": 0,
                "verdict": "",
                "tags": {},
            }
            max_diff = 0

            for tag, pop_count in pop_counts.items():
                pop_pct = pop_count / pop_total * 100 if pop_total > 0 else 0
                samp_count = sample_counts.get(tag, 0)
                samp_pct = samp_count / denominator * 100 if denominator > 0 else 0
                diff = round(samp_pct - pop_pct, 2)
                abs_diff = abs(diff)
                max_diff = max(max_diff, abs_diff)

                if abs_diff > 10:   flag = "!!!"
                elif abs_diff > 5:  flag = " ! "
                elif abs_diff > 3:  flag = " ~ "
                else:               flag = " ok"

                print(f"  {tag:<50} {pop_pct:5.1f}% {samp_pct:5.1f}% "
                      f"{diff:+6.1f}%  {flag}  {samp_count:>4}/{denominator}")

                dim_result["tags"][tag] = {
                    "pop_pct": round(pop_pct, 2),
                    "samp_pct": round(samp_pct, 2),
                    "diff": diff,
                    "count": samp_count,
                    "denominator": denominator,
                }

            if max_diff <= 3:    verdict = "GOOD"
            elif max_diff <= 5:  verdict = "ACCEPTABLE"
            elif max_diff <= 10: verdict = "WATCH"
            else:                verdict = "PROBLEM"

            dim_result["worst_diff"] = round(max_diff, 2)
            dim_result["verdict"] = verdict
            fandom_result["dimensions"][dim_name] = dim_result

            cov = ""
            if not is_single:
                cp = matched_total / raw_total * 100 if raw_total > 0 else 0
                cov = f"  (top N coverage: {matched_total}/{raw_total} = {cp:.0f}%)"

            print(f"  >> {dim_name.upper()} verdict: {verdict} "
                  f"(worst: {max_diff:.1f} pp){cov}")

        results[fandom_name] = fandom_result

    # Legend
    print(f"\n{'=' * 70}")
    print("LEGEND")
    print(f"{'=' * 70}")
    print("   ok  = within 3 pp  (representative)")
    print("    ~  = 3 to 5 pp    (acceptable)")
    print("    !  = 5 to 10 pp   (notable deviation)")
    print("   !!! = >10 pp       (sample may not be representative)")
    print()
    print("Multi tag fields use ONLY matched tags as denominator")
    print("(apples to apples with population numbers).")

    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    summary_path = os.path.abspath(ARGS.summary)
    json_path = os.path.abspath(ARGS.json_out)

    # Tee stdout to checker_summary.txt
    summary_file = open(summary_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, summary_file)

    try:
        results = run_check(CSV_PATH, TAG_SEP)
    finally:
        sys.stdout = old_stdout
        summary_file.close()

    # Write structured JSON for the fixer
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSummary written to: {summary_path}")
    print(f"JSON data written to: {json_path}")


if __name__ == "__main__":
    main()

