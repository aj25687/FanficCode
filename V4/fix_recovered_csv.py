#!/usr/bin/env python3
"""
Convert previously recovered AO3 CSV rows into the normal sampler CSV format.

This is for recovered rows that currently look like:

    sampling_mode=cache_recovery
    fandom_scope=cache_recovery
    observation_key=CACHE_RECOVERY|...

The user confirmed that all runs so far used the same settings, so this script
uses those exact settings for every converted row:

    fandom_scope      = same
    fandom            = Harry Potter - J. K. Rowling
    category_filter   = 0  (All)
    warning_filter    = 0  (All)
    crossover_filter  = 1  (No crossovers)
    chapter_filter    = 0  (All)
    completion_filter = 0  (All)
    timeframe_filter  = 0  (All time)
    sampling_mode     = all
    sampling_seed     = 0

For the known contaminated rows, date_published is intentionally copied from
date_updated.

The script preserves the actual run_timestamp, sampling_date, sampled_page,
and all cached work metadata.

Usage:

    python fix_recovered_csv.py

Defaults:
    input  = recovered_fics.csv
    output = recovered_fics_fixed.csv

Or:

    python fix_recovered_csv.py --input recovered_fics.csv \
        --output ao3_results_fixed.csv

The original input file is never overwritten unless --in-place is used.
"""

import argparse
import csv
import os
import sys
from pathlib import Path


# ============================================================
# Fixed research settings confirmed by the user
# ============================================================

FIXED_FANDOM_SCOPE = "same"
FIXED_FANDOM = "Harry Potter - J. K. Rowling"
FIXED_CATEGORY_FILTER = "0"
FIXED_WARNING_FILTER = "0"
FIXED_CROSSOVER_FILTER = "1"
FIXED_CHAPTER_FILTER = "0"
FIXED_COMPLETION_FILTER = "0"
FIXED_TIMEFRAME_FILTER = "0"
FIXED_SAMPLING_MODE = "all"
FIXED_SAMPLING_SEED = "0"


CSV_FIELDS = [
    "run_timestamp",
    "sampling_date",
    "role",
    "observation_key",
    "work_id",
    "title",
    "fandom",
    "fandoms",
    "relationships",
    "fandom_scope",
    "category_filter",
    "warning_filter",
    "crossover_filter",
    "chapter_filter",
    "completion_filter",
    "timeframe_filter",
    "sampling_mode",
    "sampling_seed",
    "sampled_page",
    "hits",
    "kudos",
    "bookmarks",
    "comments",
    "words",
    "chapters",
    "rating",
    "warnings",
    "category",
    "language",
    "date_published",
    "date_updated",
    "kudos_to_hits",
    "bookmarks_to_hits",
    "comments_to_hits",
    "comments_to_kudos",
    "kudos_per_10k_words",
    "bookmarks_per_10k_words",
    "comments_per_10k_words",
    "is_text_work",
    "is_crossover",
]


def normalize_work_id(value):
    """Return a clean work ID string."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]

    return text


def make_observation_key(work_id):
    """
    Recreate the normal sampler observation_key format.

    This matches the format produced by the main sampler:
        work_id|fandom_scope|category|warning|crossover|
        chapter|completion|timeframe|sampling_mode|sampling_seed
    """
    return "|".join([
        work_id,
        FIXED_FANDOM_SCOPE,
        FIXED_CATEGORY_FILTER,
        FIXED_WARNING_FILTER,
        FIXED_CROSSOVER_FILTER,
        FIXED_CHAPTER_FILTER,
        FIXED_COMPLETION_FILTER,
        FIXED_TIMEFRAME_FILTER,
        FIXED_SAMPLING_MODE,
        FIXED_SAMPLING_SEED,
    ])


def parse_bool(value):
    """
    Normalize common CSV boolean representations.

    Returns the original value when it is not obviously boolean, so we do not
    silently destroy unexpected data.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return "True"

    if text in {"false", "0", "no"}:
        return "False"

    return str(value)


def convert_row(old_row):
    """
    Convert one recovered row into the normal sampler format.
    """
    work_id = normalize_work_id(old_row.get("work_id"))
    date_updated = old_row.get("date_updated", "").strip()

    new_row = {field: "" for field in CSV_FIELDS}

    new_row.update({
        "run_timestamp": old_row.get("run_timestamp", ""),
        "sampling_date": old_row.get("sampling_date", ""),
        "role": old_row.get("role", "cohort") or "cohort",
        "observation_key": make_observation_key(work_id),
        "work_id": work_id,
        "title": old_row.get("title", ""),
        "fandom": FIXED_FANDOM,
        "fandoms": old_row.get("fandoms", "") or FIXED_FANDOM,
        "relationships": old_row.get("relationships", ""),
        "fandom_scope": FIXED_FANDOM_SCOPE,
        "category_filter": FIXED_CATEGORY_FILTER,
        "warning_filter": FIXED_WARNING_FILTER,
        "crossover_filter": FIXED_CROSSOVER_FILTER,
        "chapter_filter": FIXED_CHAPTER_FILTER,
        "completion_filter": FIXED_COMPLETION_FILTER,
        "timeframe_filter": FIXED_TIMEFRAME_FILTER,
        "sampling_mode": FIXED_SAMPLING_MODE,
        "sampling_seed": FIXED_SAMPLING_SEED,
        "sampled_page": old_row.get("sampled_page", ""),
        "hits": old_row.get("hits", ""),
        "kudos": old_row.get("kudos", ""),
        "bookmarks": old_row.get("bookmarks", ""),
        "comments": old_row.get("comments", ""),
        "words": old_row.get("words", ""),
        "chapters": old_row.get("chapters", ""),
        "rating": old_row.get("rating", ""),
        "warnings": old_row.get("warnings", ""),
        "category": old_row.get("category", ""),
        "language": old_row.get("language", ""),
        # User explicitly requested that the contaminated rows use
        # date_updated as date_published.
        "date_published": date_updated,
        "date_updated": date_updated,
        "kudos_to_hits": old_row.get("kudos_to_hits", ""),
        "bookmarks_to_hits": old_row.get("bookmarks_to_hits", ""),
        "comments_to_hits": old_row.get("comments_to_hits", ""),
        "comments_to_kudos": old_row.get("comments_to_kudos", ""),
        "kudos_per_10k_words": old_row.get("kudos_per_10k_words", ""),
        "bookmarks_per_10k_words": old_row.get("bookmarks_per_10k_words", ""),
        "comments_per_10k_words": old_row.get("comments_per_10k_words", ""),
        "is_text_work": parse_bool(old_row.get("is_text_work", "")),
        "is_crossover": parse_bool(old_row.get("is_crossover", "")),
    })

    return new_row


def read_rows(path):
    """Read all input rows and validate the basic required fields."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        required = {"work_id", "title", "date_updated"}
        missing = sorted(required - set(reader.fieldnames))

        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(missing)
            )

        return list(reader)


def write_rows(path, rows):
    """Write the converted CSV atomically."""
    temp_path = path.with_name(path.name + ".tmp")

    try:
        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

            file.flush()
            os.fsync(file.fileno())

        temp_path.replace(path)

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert recovered AO3 CSV rows into the normal sampler format."
    )

    parser.add_argument(
        "--input",
        default="recovered_fics.csv",
        help="Input recovered CSV (default: recovered_fics.csv)",
    )

    parser.add_argument(
        "--output",
        default="recovered_fics_fixed.csv",
        help="Output CSV (default: recovered_fics_fixed.csv)",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input CSV with the converted version.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.in_place:
        output_path = input_path
    else:
        output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        return 1

    if not args.in_place and input_path.resolve() == output_path.resolve():
        print("ERROR: Input and output are the same file. Use --in-place instead.")
        return 1

    print("AO3 Recovered CSV Formatter")
    print("===========================")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    try:
        old_rows = read_rows(input_path)
    except Exception as error:
        print(f"ERROR: Could not read input CSV: {error}")
        return 1

    print(f"Input rows: {len(old_rows)}")

    converted_rows = []
    seen_work_ids = set()
    duplicate_work_ids = 0
    blank_work_ids = 0

    for old_row in old_rows:
        work_id = normalize_work_id(old_row.get("work_id"))

        if not work_id:
            blank_work_ids += 1
            continue

        if work_id in seen_work_ids:
            duplicate_work_ids += 1
            continue

        seen_work_ids.add(work_id)
        converted_rows.append(convert_row(old_row))

    try:
        write_rows(output_path, converted_rows)
    except Exception as error:
        print(f"ERROR: Could not write output CSV: {error}")
        return 1

    print(f"Converted rows: {len(converted_rows)}")
    print(f"Duplicate work IDs removed: {duplicate_work_ids}")
    print(f"Rows with blank work IDs skipped: {blank_work_ids}")
    print()
    print("Fixed settings applied:")
    print(f"  fandom:             {FIXED_FANDOM}")
    print(f"  fandom_scope:       {FIXED_FANDOM_SCOPE}")
    print(f"  category_filter:    {FIXED_CATEGORY_FILTER}")
    print(f"  warning_filter:     {FIXED_WARNING_FILTER}")
    print(f"  crossover_filter:   {FIXED_CROSSOVER_FILTER}")
    print(f"  chapter_filter:     {FIXED_CHAPTER_FILTER}")
    print(f"  completion_filter:  {FIXED_COMPLETION_FILTER}")
    print(f"  timeframe_filter:   {FIXED_TIMEFRAME_FILTER}")
    print(f"  sampling_mode:      {FIXED_SAMPLING_MODE}")
    print(f"  sampling_seed:      {FIXED_SAMPLING_SEED}")
    print("  date_published:     copied from date_updated")
    print()
    print(f"Done. Wrote {len(converted_rows)} row(s) to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
