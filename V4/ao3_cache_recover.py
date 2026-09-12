#!/usr/bin/env python3
"""
AO3 cache -> CSV recovery utility.

Purpose:
    Recover work records that were successfully saved in ao3_cache.json
    but never made it into ao3_results.csv because a run was interrupted.

Behavior:
    1. Loads every cached search-result page from ao3_cache.json.
    2. Compares cached work IDs against work IDs already present in ao3_results.csv.
    3. Adds missing cached works to the CSV.
    4. Uses the exact fixed research settings the user has confirmed were used
       for every run so far.
    5. Copies date_updated into date_published, per the current research workflow.
    6. After a successful CSV write, clears the cache to:
           {"pages": {}, "page_counts": {}, "metadata": {}}

Important:
    The cache contains search-result metadata, not the full work page. Therefore
    date_published is intentionally set equal to date_updated for recovered rows.

Usage:
    python ao3_cache_recover.py

Optional:
    python ao3_cache_recover.py --dry-run
    python ao3_cache_recover.py --cache path/to/cache.json --csv path/to/results.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse


CACHE_FILE = "ao3_cache.json"
CSV_FILE = "ao3_results.csv"

# Keep this exactly aligned with ao3_research_sampler_v4.py.
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

CATEGORY_ID_TO_FILTER = {
    "23": "1",     # M/M
    "116": "2",    # F/F
    "22": "3",     # F/M
    "2246": "4",  # Multi
    "21": "5",     # Gen
}

WARNING_ID_TO_FILTER = {
    "16": "1",  # No Archive Warnings Apply
    "14": "2",  # Creator Chose Not To Use Archive Warnings
    "17": "3",  # Graphic Depictions Of Violence
    "18": "4",  # Major Character Death
    "19": "5",  # Rape/Non-Con
    "20": "6",  # Underage
}

# These are the fixed settings the user has used for every run so far.
FIXED_FANDOM_SCOPE = "same"
FIXED_FANDOM = "Harry Potter - J. K. Rowling"
FIXED_CATEGORY_FILTER = "0"
FIXED_WARNING_FILTER = "0"
FIXED_CROSSOVER_FILTER = "1"   # No crossovers
FIXED_CHAPTER_FILTER = "0"     # All chapter counts
FIXED_COMPLETION_FILTER = "0"  # All completion states
FIXED_TIMEFRAME_FILTER = "0"   # All time
FIXED_SAMPLING_MODE = "all"
FIXED_SAMPLING_SEED = "0"



def empty_cache():
    return {
        "pages": {},
        "page_counts": {},
        "metadata": {},
    }


def normalize_work_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_cache(path):
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        cache = json.load(file)

    if not isinstance(cache, dict):
        raise ValueError("Cache JSON must contain an object at the top level.")

    pages = cache.get("pages", {})
    if not isinstance(pages, dict):
        raise ValueError("Cache 'pages' must be a JSON object.")

    return cache


def load_existing_work_ids(path):
    """
    Read work IDs already represented in the CSV.

    This intentionally uses work_id, not observation_key, because the recovery
    utility's job is to recover physical works that were cached but never
    reached the CSV at all. A work already present in the CSV is not duplicated.
    """
    existing_ids = set()

    if not path.exists():
        return existing_ids

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            work_id = normalize_work_id(row.get("work_id"))
            if work_id:
                existing_ids.add(work_id)

    return existing_ids


def first_query_value(query, key):
    values = query.get(key)
    if not values:
        return ""
    return values[0]


def parse_cached_url(url):
    """
    Recover only the filters explicitly encoded in the cached AO3 URL.

    We deliberately do not guess values that are not present in the URL.
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    fandom = first_query_value(query, "work_search[fandom_names]")

    category_ids = query.get("work_search[category_ids][]", [])
    category_filter = "0"
    for category_id in category_ids:
        if category_id in CATEGORY_ID_TO_FILTER:
            category_filter = CATEGORY_ID_TO_FILTER[category_id]
            break

    warning_ids = query.get("work_search[archive_warning_ids][]", [])
    warning_filter = "0"
    for warning_id in warning_ids:
        if warning_id in WARNING_ID_TO_FILTER:
            warning_filter = WARNING_ID_TO_FILTER[warning_id]
            break

    crossover = first_query_value(query, "work_search[crossover]")
    if crossover == "F":
        crossover_filter = "1"
    elif crossover == "T":
        crossover_filter = "2"
    else:
        crossover_filter = "0"

    complete = first_query_value(query, "work_search[complete]")
    completion_filter = "1" if complete == "T" else "0"

    date_from = first_query_value(query, "work_search[date_from]")
    date_to = first_query_value(query, "work_search[date_to]")

    if date_from or date_to:
        timeframe_filter = f"4:{date_from}:{date_to}"
    else:
        timeframe_filter = "0"

    page_value = first_query_value(query, "page")
    try:
        sampled_page = int(page_value) if page_value else ""
    except ValueError:
        sampled_page = ""

    return {
        "fandom": fandom,
        "category_filter": category_filter,
        "warning_filter": warning_filter,
        "crossover_filter": crossover_filter,
        "chapter_filter": "0",
        "completion_filter": completion_filter,
        "timeframe_filter": timeframe_filter,
        "sampled_page": sampled_page,
    }


def make_recovery_observation_key(work_id):
    """Create the same observation-key format used by the normal sampler.

    The user has confirmed that every run so far used the same research
    settings, so recovered rows can use the normal observation-key format
    rather than a special recovery-only key.
    """
    values = [
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
    ]
    return "|".join(values)


def make_recovered_row(work, url, run_timestamp):
    """Convert one cached search-result record into the normal CSV schema.

    The user confirmed that all runs to date used these exact settings:
      fandom = Harry Potter - J. K. Rowling
      category = All
      warning = All
      crossover = No crossovers
      chapters = All
      completion = All
      timeframe = All time
      sampling mode = all
      sampling seed = 0

    The cache's date_updated value is intentionally copied into date_published
    for now, per the user's instruction.
    """
    work_id = normalize_work_id(work.get("work_id"))
    words = int(work.get("words", 0) or 0)
    date_updated = work.get("date_updated", "")

    row = {field: "" for field in CSV_FIELDS}

    row.update({
        "run_timestamp": run_timestamp,
        "sampling_date": datetime.now(timezone.utc).date().isoformat(),
        "role": "cohort",
        "observation_key": make_recovery_observation_key(work_id),
        "work_id": work_id,
        "title": work.get("title", ""),
        "fandom": FIXED_FANDOM,
        "fandoms": work.get("fandoms", FIXED_FANDOM),
        "relationships": work.get("relationships", ""),
        "fandom_scope": FIXED_FANDOM_SCOPE,
        "category_filter": FIXED_CATEGORY_FILTER,
        "warning_filter": FIXED_WARNING_FILTER,
        "crossover_filter": FIXED_CROSSOVER_FILTER,
        "chapter_filter": FIXED_CHAPTER_FILTER,
        "completion_filter": FIXED_COMPLETION_FILTER,
        "timeframe_filter": FIXED_TIMEFRAME_FILTER,
        "sampling_mode": FIXED_SAMPLING_MODE,
        "sampling_seed": FIXED_SAMPLING_SEED,
        "sampled_page": parse_cached_url(url)["sampled_page"],
        "hits": work.get("hits", 0),
        "kudos": work.get("kudos", 0),
        "bookmarks": work.get("bookmarks", 0),
        "comments": work.get("comments", 0),
        "words": words,
        "chapters": work.get("chapters", 1),
        "rating": work.get("rating", ""),
        "warnings": work.get("warnings", ""),
        "category": work.get("category", ""),
        "language": work.get("language", ""),
        "date_published": date_updated,
        "date_updated": date_updated,
        "kudos_to_hits": work.get("kudos_to_hits", 0.0),
        "bookmarks_to_hits": work.get("bookmarks_to_hits", 0.0),
        "comments_to_hits": work.get("comments_to_hits", 0.0),
        "comments_to_kudos": work.get("comments_to_kudos", 0.0),
        "kudos_per_10k_words": work.get("kudos_per_10k_words", 0.0),
        "bookmarks_per_10k_words": work.get("bookmarks_per_10k_words", 0.0),
        "comments_per_10k_words": work.get("comments_per_10k_words", 0.0),
        "is_text_work": words > 0,
        "is_crossover": bool(work.get("is_crossover", False)),
    })

    return row


def collect_missing_rows(cache, existing_ids):
    """
    Collect at most one recovered row per physical work ID.

    A work can appear in more than one cached page. Prefer the first occurrence,
    which is enough for recovery because the cached record itself contains the
    same work metadata regardless of which matching search page contained it.
    """
    pages = cache.get("pages", {})
    recovered_rows = []
    seen_cache_ids = set()
    duplicate_cache_records = 0
    invalid_records = 0

    for url, page_results in pages.items():
        if not isinstance(page_results, list):
            continue

        for work in page_results:
            if not isinstance(work, dict):
                invalid_records += 1
                continue

            work_id = normalize_work_id(work.get("work_id"))
            if not work_id:
                invalid_records += 1
                continue

            if work_id in existing_ids:
                continue

            if work_id in seen_cache_ids:
                duplicate_cache_records += 1
                continue

            seen_cache_ids.add(work_id)
            recovered_rows.append((url, work))

    return recovered_rows, duplicate_cache_records, invalid_records


def write_rows(path, rows):
    """
    Append rows atomically at the CSV level.

    A temporary file is used so a failed write does not destroy the original
    CSV. Existing rows are copied verbatim, then recovered rows are appended.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    temp_path = path.with_name(path.name + ".recovery_tmp")

    try:
        existing_rows = []
        fieldnames = CSV_FIELDS

        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_rows = list(reader)

                if reader.fieldnames:
                    missing_fields = [
                        field for field in CSV_FIELDS if field not in reader.fieldnames
                    ]
                    if missing_fields:
                        raise ValueError(
                            "CSV is missing required fields: "
                            + ", ".join(missing_fields)
                        )

        with temp_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for old_row in existing_rows:
                writer.writerow({
                    field: old_row.get(field, "")
                    for field in fieldnames
                })

            for new_row in rows:
                writer.writerow(new_row)

            file.flush()
            os.fsync(file.fileno())

        temp_path.replace(path)

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def clear_cache(path):
    """
    Replace the cache with a tiny empty cache.

    The old cache is NOT deleted until after CSV writing succeeds.
    """
    temp_path = path.with_name(path.name + ".clear_tmp")

    empty = empty_cache()

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(empty, file)
            file.write("\n")
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
        description="Recover AO3 works from ao3_cache.json into ao3_results.csv."
    )
    parser.add_argument(
        "--cache",
        default=CACHE_FILE,
        help=f"Cache JSON path (default: {CACHE_FILE})",
    )
    parser.add_argument(
        "--csv",
        default=CSV_FILE,
        help=f"Results CSV path (default: {CSV_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recovered without changing the CSV or cache.",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)
    csv_path = Path(args.csv)

    print("AO3 Cache Recovery")
    print("==================")
    print(f"Cache: {cache_path}")
    print(f"CSV:   {csv_path}")
    print()

    try:
        cache = load_cache(cache_path)
    except Exception as error:
        print(f"ERROR: Could not load cache: {error}")
        return 1

    pages = cache.get("pages", {})
    page_counts = cache.get("page_counts", {})

    print(f"Cached search pages: {len(pages)}")
    print(f"Cached page counts:  {len(page_counts)}")

    try:
        existing_ids = load_existing_work_ids(csv_path)
    except Exception as error:
        print(f"ERROR: Could not read existing CSV: {error}")
        return 1

    print(f"Work IDs already in CSV: {len(existing_ids)}")

    recovered_records, duplicate_cache_records, invalid_records = (
        collect_missing_rows(cache, existing_ids)
    )

    print(f"Unique missing cached works: {len(recovered_records)}")
    print(f"Duplicate cached records ignored: {duplicate_cache_records}")
    print(f"Invalid cached records ignored: {invalid_records}")
    print()

    if recovered_records:
        print("First 10 recovered work IDs:")
        for url, work in recovered_records[:10]:
            print(
                f"  {normalize_work_id(work.get('work_id'))}: "
                f"{work.get('title', '')}"
            )
        if len(recovered_records) > 10:
            print(f"  ... and {len(recovered_records) - 10} more")
    else:
        print("There are no cached works missing from the CSV.")

    if args.dry_run:
        print()
        print("DRY RUN: no files were changed.")
        return 0

    if recovered_records:
        run_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            make_recovered_row(work, url, run_timestamp)
            for url, work in recovered_records
        ]

        try:
            write_rows(csv_path, rows)
        except Exception as error:
            print()
            print(f"ERROR: CSV recovery failed: {error}")
            print("The cache was NOT cleared.")
            return 1

        print()
        print(f"Added {len(rows)} recovered work(s) to {csv_path}.")

    # Only clear after the CSV step has succeeded.
    try:
        clear_cache(cache_path)
    except Exception as error:
        print()
        print(f"ERROR: CSV recovery succeeded, but cache clearing failed: {error}")
        print("Your data should be safe in the CSV. Do not rerun blindly until")
        print("you check the CSV, because the old cache may still be present.")
        return 1

    print(f"Cleared {cache_path}.")
    print("Cache is now small and ready for the next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
