#!/usr/bin/env python3
"""
sample_checker.py
=================
Compares your hand-coding sample CSV against known AO3 fandom population
numbers to verify the sample is proportionally representative.

Checks: ratings, warnings, categories, top 10 characters, top 10 relationships.

For multi-tag fields (warnings, categories, characters, relationships),
the comparison uses ONLY tags that match the known top-N list so the
percentages are apples-to-apples with the population numbers.

EVERYTHING that is printed is also written to checker_summary.txt. The file is
OVERWRITTEN on every run (never appended to).

Usage:
    python3 sample_checker.py
    python3 sample_checker.py --csv path/to/ao3_testing_sample.csv
    python3 sample_checker.py --out some_other_summary.txt

Also importable (ao3_fix_sample.py does this):
    analyze_rows(rows, sep)      -> structured results (flags, diffs, verdicts)
    dimension_stats(...)         -> the one place deviations/flags are computed
    run_check(csv, sep, out)     -> read + analyse + print + write summary file
"""

import argparse
import csv
import os
import sys
from contextlib import contextmanager

DEFAULT_CSV = "ao3_testing_sample.csv"
DEFAULT_SEP = " | "
DEFAULT_SUMMARY = "checker_summary.txt"


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
        "fandom_keywords": ["arcane"],   # NOT "league of legends": that also matches the video game
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
        "fandom_keywords": ["last airbender"],   # NOT "avatar"/"atla": those also match Legend of Korra, Avatar (2009), Atlantis...
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
# LIVE POPULATION NUMBERS
# ══════════════════════════════════════════════════════════════
# ao3_fix_sample.py reads each fandom's AO3 filter sidebar at the start of a run and writes the counts to
# population_live.json. If that file exists, its counts REPLACE the hard-coded numbers above (only for tags
# this checker already tracks), so flags are always judged against the most recent AO3 numbers.

POPULATION_FILE = "population_live.json"
POPULATION_INFO = {"fetched_at": None, "source": None}
_POP_KEYS = ("ratings", "warnings", "categories", "characters", "relationships")


def apply_population(data):
    """data = {"fetched_at": iso, "fandoms": {name: {"total": n, "ratings": {tag: n}, ...}}}"""
    changed = 0
    for name, d in data.get("fandoms", {}).items():
        if name not in FANDOMS:
            continue
        if d.get("total"):
            FANDOMS[name]["total"] = int(d["total"])
        for key in _POP_KEYS:
            for tag, count in d.get(key, {}).items():
                if tag in FANDOMS[name][key] and count:
                    if FANDOMS[name][key][tag] != int(count):
                        changed += 1
                    FANDOMS[name][key][tag] = int(count)
    POPULATION_INFO["fetched_at"] = data.get("fetched_at")
    POPULATION_INFO["source"] = data.get("source")
    return changed


def load_live_population(path=None):
    """Apply population_live.json if present (current folder first, then next to this script)."""
    import json
    candidates = [path] if path else [os.path.join(os.getcwd(), POPULATION_FILE),
                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), POPULATION_FILE)]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as fh:
                    apply_population(json.load(fh))
                return True
            except (OSError, ValueError):
                pass
    return False


load_live_population()


# (heading, csv column, key inside FANDOMS[...], is_single_value)
DIMENSIONS = [
    ("RATINGS",       "rating",        "ratings",       True),
    ("WARNINGS",      "warnings",      "warnings",      False),
    ("CATEGORIES",    "category",      "categories",    False),
    ("CHARACTERS",    "characters",    "characters",    False),
    ("RELATIONSHIPS", "relationships", "relationships", False),
]


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


def split_tags(value, sep=DEFAULT_SEP):
    if not value:
        return []
    return [t.strip() for t in value.split(sep) if t.strip()]


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


def count_matched_tags(rows, field, pop_tags, sep=DEFAULT_SEP):
    """Count sample tags matching population tags.
    Returns:
      counts       = {pop_tag: n_matches}
      matched_total = number of tag occurrences that matched ANY known tag
                      (this is the correct denominator for apples-to-apples comparison)
      raw_total     = total tag occurrences (for the coverage stat)
    """
    pop_tags = list(pop_tags)
    counts = {tag: 0 for tag in pop_tags}
    matched_total = 0
    raw_total = 0
    for row in rows:
        tags = split_tags(row.get(field, ""), sep)
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
# DEVIATIONS, FLAGS, VERDICTS  (single source of truth)
# ══════════════════════════════════════════════════════════════

def flag_for(abs_diff):
    if abs_diff > 10:
        return "!!!"
    if abs_diff > 5:
        return " ! "
    if abs_diff > 3:
        return " ~ "
    return " ok"


def verdict_for(max_diff):
    if max_diff <= 3:
        return "GOOD"
    if max_diff <= 5:
        return "ACCEPTABLE"
    if max_diff <= 10:
        return "WATCH"
    return "PROBLEM"


def is_flagged(flag):
    """True for the two flags that mean 'notable deviation': ! and !!!"""
    return flag.strip() in ("!", "!!!")


def dimension_stats(counts, denominator, pop_counts):
    """Compare sample counts to population counts for ONE dimension.
    `denominator` is the number of works (ratings) or the number of matched
    tag slots (multi-tag fields). Returns per-tag rows plus the verdict."""
    pop_total = sum(pop_counts.values())
    tags = []
    max_diff = 0.0
    for tag, pop_count in pop_counts.items():
        pop_pct = pop_count / pop_total * 100 if pop_total > 0 else 0
        samp_count = counts.get(tag, 0)
        samp_pct = samp_count / denominator * 100 if denominator > 0 else 0
        diff = samp_pct - pop_pct
        abs_diff = abs(diff)
        max_diff = max(max_diff, abs_diff)
        tags.append({
            "tag": tag, "pop_pct": pop_pct, "samp_pct": samp_pct, "diff": diff,
            "abs_diff": abs_diff, "flag": flag_for(abs_diff), "count": samp_count,
        })
    return {"tags": tags, "max_diff": max_diff, "verdict": verdict_for(max_diff),
            "denominator": denominator}


def analyze_rows(rows, sep=DEFAULT_SEP):
    """Group rows by fandom and run every dimension. No printing."""
    fandom_rows, unknown = {}, []
    for row in rows:
        fandom = detect_fandom(row)
        if fandom == "Unknown":
            unknown.append(row)
        else:
            fandom_rows.setdefault(fandom, []).append(row)

    analysis = {"n_total": len(rows), "unknown": len(unknown),
                "counts": {name: len(fandom_rows.get(name, [])) for name in FANDOMS},
                "fandoms": {}}

    for fandom_name, info in FANDOMS.items():
        f_rows = fandom_rows.get(fandom_name, [])
        if not f_rows:
            continue
        n = len(f_rows)
        dims = {}
        for dim_name, field, pop_key, is_single in DIMENSIONS:
            pop_counts = info[pop_key]
            counts, matched_total, raw_total = count_matched_tags(f_rows, field, pop_counts.keys(), sep)
            if is_single:
                denominator = n
            else:
                denominator = matched_total if matched_total > 0 else 1
            stats = dimension_stats(counts, denominator, pop_counts)
            stats.update({"matched_total": matched_total, "raw_total": raw_total,
                          "is_single": is_single, "field": field})
            dims[dim_name] = stats
        analysis["fandoms"][fandom_name] = {"n": n, "dims": dims}
    return analysis


def flagged_items(analysis):
    """Every (fandom, dimension, tag) whose flag is ! or !!!, worst first."""
    out = []
    for fandom, fa in analysis["fandoms"].items():
        for dim_name, ds in fa["dims"].items():
            for t in ds["tags"]:
                if is_flagged(t["flag"]):
                    out.append({"fandom": fandom, "dim": dim_name, **t})
    out.sort(key=lambda x: -x["abs_diff"])
    return out


# ══════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════

def print_report(analysis, csv_name):
    print(f"Loaded {analysis['n_total']} works from {csv_name}")
    if POPULATION_INFO["fetched_at"]:
        print(f"Population numbers: live AO3 sidebar counts fetched {POPULATION_INFO['fetched_at']}")
    else:
        print("Population numbers: built-in counts (run ao3_fix_sample.py to refresh them from AO3)")
    print()

    print("=" * 70)
    print("FANDOM BREAKDOWN IN SAMPLE")
    print("=" * 70)
    for name in FANDOMS:
        n = analysis["counts"][name]
        pop = FANDOMS[name]["total"]
        print(f"  {name:>15}: {n:>4} sampled  /  {pop:>7,} population")
    if analysis["unknown"]:
        print(f"  {'Unknown':>15}: {analysis['unknown']:>4} sampled  (no population data)")
    print(f"  {'TOTAL':>15}: {analysis['n_total']:>4} sampled")

    for fandom_name, fandom_info in FANDOMS.items():
        fa = analysis["fandoms"].get(fandom_name)
        if fa is None:
            print(f"\n{'=' * 70}")
            print(f"SKIPPING {fandom_name}: no works in sample")
            continue

        n = fa["n"]
        print(f"\n{'=' * 70}")
        print(f"{fandom_name.upper()}  ({n} sampled  /  {fandom_info['total']:,} population)")
        print(f"{'=' * 70}")

        for dim_name, ds in fa["dims"].items():
            print(f"\n  {dim_name}")
            print(f"  {'-' * 66}")
            print(f"  {'Tag':<50} {'Pop%':>6} {'Samp%':>6} {'Diff':>7}  {'':>3}  {'Count':>8}")
            for t in ds["tags"]:
                print(f"  {t['tag']:<50} {t['pop_pct']:5.1f}% {t['samp_pct']:5.1f}% "
                      f"{t['diff']:+6.1f}%  {t['flag']}  {t['count']:>4}/{ds['denominator']}")
            print(f"  >> {dim_name} verdict: {ds['verdict']} (worst deviation: {ds['max_diff']:.1f} pp)")
            if not ds["is_single"]:
                matched, raw = ds["matched_total"], ds["raw_total"]
                coverage_pct = matched / raw * 100 if raw > 0 else 0
                print(f"    (top-N tags cover {matched} of {raw} "
                      f"tag slots = {coverage_pct:.0f}% coverage)")

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


# ══════════════════════════════════════════════════════════════
# RUN  (prints to the terminal AND overwrites the summary file)
# ══════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


@contextmanager
def _tee_to_file(path):
    """Mirror stdout into `path`, opened in WRITE mode so old content is wiped."""
    if not path:
        yield
        return
    original = sys.stdout
    with open(path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original, fh)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stdout = original


def run_check(csv_path, sep=DEFAULT_SEP, summary_path=DEFAULT_SUMMARY):
    """Read the CSV, analyse it, print the report and overwrite `summary_path`."""
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    rows = read_csv(csv_path)
    analysis = analyze_rows(rows, sep)
    with _tee_to_file(summary_path):
        print_report(analysis, os.path.basename(csv_path))
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Check AO3 sample representativeness.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Sample CSV to check.")
    parser.add_argument("--sep", default=DEFAULT_SEP, help="Tag separator in CSV (default: ' | ').")
    parser.add_argument("--out", default=DEFAULT_SUMMARY,
                        help="Summary file (overwritten every run). Default: checker_summary.txt")
    args = parser.parse_args()
    try:
        run_check(args.csv, args.sep, args.out)
    except FileNotFoundError as err:
        sys.exit(f"ERROR: {err}")
    print(f"\n(Summary written to {args.out})")


if __name__ == "__main__":
    main()