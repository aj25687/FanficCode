import csv
import json
import os
import random
import re
import time
import urllib.parse
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "AO3StatsComparator/1.2 (Educational/Research Scraping Script)"
}

# AO3 requires this cookie or it shows a "this may contain adult content"
# interstitial page instead of the real work page for Mature/Explicit works,
# which silently produces all-zero stats.
COOKIES = {"view_adult": "true"}

CACHE_FILE = "ao3_cache.json"
CSV_FILE = "ao3_results.csv"
DIST_FILE = "ao3_distributions.json"

# Confirmed against AO3's own reference post (archiveofourown.org/admin_posts/349):
#   Categories: Gen=21, F/M=22, M/M=23, Other=24, F/F=116, Multi=2246
CATEGORY_MAP = {
    "1": "23",    # M/M
    "2": "116",   # F/F
    "3": "22",    # F/M
    "4": "2246",  # Multi
    "5": "21",    # Gen
}

# Warning IDs
WARNING_MAP = {
    "1": "16",  # No Warnings Apply
    "2": "14",  # Creator Chose Not To Use
    "3": "17",  # Graphic Violence
    "4": "18",  # Major Character Death
}

# Consecutive hard failures (403/503/525/etc.) before we stop the whole run.
# This is a "the site is telling us no, so stop" circuit breaker, not an
# evasion mechanism.
MAX_CONSECUTIVE_HARD_FAILURES = 3

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.cookies.update(COOKIES)


# ---------------------------------------------------------------------------
# Session / cookies
# ---------------------------------------------------------------------------

def parse_cookie_string(cookie_str):
    """Parses a pasted cookie value into a dict suitable for
    requests.Session.cookies.update().

    Handles the formats people actually paste:
    - A full 'Cookie:' header value: "_otwarchive_session=abc; other=xyz"
    - The same, but including the header NAME (a common result of
      right-click > Copy Value on the Cookie row in browser dev tools):
      "cookie: _otwarchive_session=abc; other=xyz" or "Cookie: ..."
    - A bare single token with no "name=value" structure at all (what you
      get copying a single cell's value from the Application/Storage tab) —
      assumed to be the _otwarchive_session value itself, since that's the
      one cookie AO3 actually needs to recognize a logged-in session.

    Rails-signed cookie VALUES (like _otwarchive_session's) routinely
    contain "=" characters of their own (base64 padding), so a bare value
    can't just be detected by "no '=' anywhere in the string" — that breaks
    on real tokens. Instead we first look for the literal, known cookie
    name "_otwarchive_session=" anywhere in the string and, if found,
    capture everything up to the next ";" as its value (correctly handling
    any "=" inside that value). Only if that anchor isn't found at all do
    we fall back to treating the whole string as a bare token.
    """
    cookie_str = cookie_str.strip().strip('"').strip("'").strip()

    # Strip a leading "Cookie:" / "cookie:" header-name prefix if present.
    cookie_str = re.sub(r"^\s*cookie\s*:\s*", "", cookie_str, flags=re.IGNORECASE)

    cookies = {}

    # Anchor on the known cookie name first, since its value legitimately
    # contains "=" characters that break naive split-on-"=" parsing.
    m = re.search(r"_otwarchive_session\s*=\s*([^;]+)", cookie_str, flags=re.IGNORECASE)
    if m:
        cookies["_otwarchive_session"] = m.group(1).strip().strip('"').strip("'")
        cookie_str = cookie_str[: m.start()] + cookie_str[m.end():]

    # Pick up any other plausible "name=value" pairs on remaining
    # ";"-separated parts. Cookie NAMES are short simple tokens (letters,
    # digits, underscore, hyphen) per RFC 6265 — real cookie names never
    # contain "/", "+", or run 60+ characters, so requiring that here stops
    # us from misreading a chunk of an unrelated base64 blob as a "name".
    name_re = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        if name_re.match(key):
            cookies[key] = value.strip().strip('"').strip("'")

    if not cookies and cookie_str:
        # Nothing recognizable as name=value at all -- assume they pasted
        # the bare _otwarchive_session value directly.
        cookies["_otwarchive_session"] = cookie_str

    return cookies


def verify_login(session_label="this cookie"):
    """Makes one lightweight request to AO3's homepage and checks for a
    logged-in indicator, so the user finds out immediately whether their
    pasted cookie actually authenticated -- rather than discovering it much
    later when a restricted work silently fails."""
    resp = get_with_retry("https://archiveofourown.org/")
    if resp is None or resp.status_code != 200:
        print("  Could not verify login (request failed) — continuing anyway.")
        return None

    # Logged-in AO3 pages include a log-out link/form; logged-out pages show
    # a login link instead.
    is_logged_in = "/users/logout" in resp.text or 'action="/users/logout"' in resp.text

    if is_logged_in:
        print(f"  Login verified: AO3 recognizes {session_label} as a logged-in session.")
    else:
        print(
            f"  Warning: AO3's homepage does NOT show a logged-in state with "
            f"{session_label}. Restricted works will likely still fail. "
            f"This usually means the cookie value was cut off, expired, or "
            f"copied in a format the parser didn't expect."
        )
    return is_logged_in


def apply_session_cookie(cookie_str):
    """Adds a user-supplied logged-in AO3 session cookie to the shared
    SESSION, so subsequent requests are authenticated and can see works
    marked 'restricted to registered users' that anonymous requests can't.
    """
    if not cookie_str:
        return
    parsed = parse_cookie_string(cookie_str)
    if not parsed:
        print("  Could not parse that cookie string — continuing without it (anonymous requests).")
        return

    SESSION.cookies.update(parsed)
    print(f"  Parsed cookie name(s): {', '.join(sorted(parsed.keys()))}")
    if "_otwarchive_session" not in parsed:
        print(
            "  Warning: no '_otwarchive_session' cookie found in what you pasted. "
            "That's the specific cookie AO3 uses to recognize a logged-in session — "
            "without it, restricted works will still be invisible."
        )
    verify_login()


# ---------------------------------------------------------------------------
# Networking: polite delays, honest failure handling (no evasion)
# ---------------------------------------------------------------------------

class HardBlockError(Exception):
    """Raised when AO3/Cloudflare has clearly told us to stop (403, 503,
    525, or repeated 429s). We surface this to the user rather than trying
    to disguise the request and push through."""


def polite_delay():
    """Randomized delay between requests. This exists to be gentler on
    AO3's servers, not to evade detection — it's strictly slower than a
    fixed short delay would be."""
    time.sleep(random.uniform(5, 10))


def get_with_retry(url, max_retries=3):
    """GET with backoff on 429 (rate limited) and on transient network
    errors (timeouts, connection resets, DNS hiccups). Distinguishes those
    from a hard block (403/503/525), which we do not try to retry past —
    see HardBlockError. Returns None if every attempt fails with a network
    error, so callers can treat that the same as a failed request."""
    resp = None
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, timeout=30)
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)
            print(
                f"  Network error ({e.__class__.__name__}: {e}). "
                f"Waiting {wait}s before retrying ({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
            resp = None
            continue

        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  Rate limited (429). Waiting {wait}s before retrying...")
            time.sleep(wait)
            continue

        if resp.status_code in (403, 503, 525):
            # These are AO3/Cloudflare explicitly denying or failing the
            # request. We don't retry-through these; the caller decides
            # whether to treat this as fatal for the whole run.
            return resp

        return resp

    return resp


# ---------------------------------------------------------------------------
# Local disk cache (page-level): avoids re-requesting pages we already have
# ---------------------------------------------------------------------------

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  Warning: couldn't read {CACHE_FILE}, starting a fresh cache.")
    return {"pages": {}}


def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError as e:
        print(f"  Warning: couldn't write cache file: {e}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_work_id_from_blurb(work_tag):
    """AO3 search result <li> tags carry id="work_1234567"."""
    tag_id = work_tag.get("id", "")
    m = re.search(r"work_(\d+)", tag_id)
    if m:
        return m.group(1)
    # Fallback: pull it from the title link href.
    link = work_tag.select_one("h4.heading a[href^='/works/']")
    if link:
        m = re.search(r"/works/(\d+)", link.get("href", ""))
        if m:
            return m.group(1)
    return None


def extract_work_id_from_url(url):
    m = re.search(r"/works/(\d+)", url)
    return m.group(1) if m else None


def parse_ao3_work(work_url):
    """Fetches details and stats for a target AO3 work (one request)."""
    if "/chapters/" in work_url:
        work_url = work_url.split("/chapters/")[0]

    resp = get_with_retry(work_url)
    if resp is None:
        raise HardBlockError(
            "Could not reach AO3 after several retries (network error or "
            "timeout each time). Check your connection, or AO3 may be "
            "temporarily unreachable — wait a bit and try again."
        )
    if resp.status_code in (403, 503, 525):
        raise HardBlockError(
            f"AO3 returned HTTP {resp.status_code} for the target work — "
            f"likely a temporary block. Wait a while before trying again."
        )
    if resp.status_code != 200:
        raise Exception(f"Could not load work page. HTTP Status: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
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
        "work_id": extract_work_id_from_url(work_url),
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
    """Constructs an AO3 work-search filter URL. Always uses /works/search
    (never /tags/<name>/works, which requires AO3's own special escaping for
    periods/slashes/ampersands in tag names and 404s otherwise)."""
    base_url = "https://archiveofourown.org/works/search?"

    params = [
        ("utf8", "✓"),
        ("commit", "Sort and Filter"),
        ("page", str(page)),
    ]

    if fandom:
        params.append(("work_search[fandom_names]", f'"{fandom}"'))

    # Sorting is neutral (last-updated); scrape_fandom_cohort walks every
    # page of the filtered set, so sort order doesn't bias the cohort.
    params.append(("work_search[sort_column]", "revised_at"))
    params.append(("work_search[sort_direction]", "desc"))

    if category in CATEGORY_MAP:
        params.append(("work_search[category_ids][]", CATEGORY_MAP[category]))

    if warning in WARNING_MAP:
        params.append(("work_search[archive_warning_ids][]", WARNING_MAP[warning]))

    # Crossover filter: AO3 expects "true"/"false".
    if crossover == "1":
        params.append(("work_search[crossover]", "false"))  # exclude crossovers
    elif crossover == "2":
        params.append(("work_search[crossover]", "true"))   # crossovers only

    if complete_only == "1":
        params.append(("work_search[complete]", "1"))

    if timeframe in ("1", "2", "3"):
        days_map = {"1": 7, "2": 30, "3": 365}
        cutoff = datetime.today() - timedelta(days=days_map[timeframe])
        params.append(("work_search[date_from]", cutoff.strftime("%Y-%m-%d")))

    return base_url + urllib.parse.urlencode(params)


def get_result_page_count(fandom, category, warning, crossover, complete_only, timeframe, cache):
    """Finds the total number of result pages for these filters. If we've
    already learned this from a previous run (cache["page_counts"]), reuses
    that instead of making a request — handy when you're about to scan a
    later page range (e.g. 200-300) and don't need to re-check page 1
    first. Otherwise fetches page 1, reads AO3's pagination controls, and
    opportunistically caches page 1's own blurb data too, since we already
    have it in hand."""
    url = build_tag_url(fandom, category, warning, crossover, complete_only, timeframe, page=1)

    cached_count = cache.get("page_counts", {}).get(url)
    if cached_count is not None:
        print(f"  Using cached page count for these filters: {cached_count} page(s) (no request made).")
        return cached_count, url

    resp = get_with_retry(url)

    if resp is None or resp.status_code != 200:
        return 0, url

    soup = BeautifulSoup(resp.text, "html.parser")
    if not soup.select("li.work.blurb"):
        return 0, url

    page_links = soup.select("ol.pagination li a")
    page_numbers = [int(a.text) for a in page_links if a.text.strip().isdigit()]
    total_pages = max(page_numbers) if page_numbers else 1

    cache.setdefault("page_counts", {})[url] = total_pages
    cache.setdefault("pages", {})[url] = parse_blurb_page(resp.text)
    save_cache(cache)

    return total_pages, url


def parse_blurb_page(html):
    """Extracts work_id + stats from every work blurb on a search results
    page. Pure metadata extraction — no extra requests per work."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for work in soup.select("li.work.blurb"):
        stats = work.find("dl", class_="stats")
        if not stats:
            continue

        def get_val(cls):
            el = stats.find("dd", class_=cls)
            return int(el.text.replace(",", "")) if el and el.text.strip() else 0

        ch_node = stats.find("dd", class_="chapters")
        chapters = 1
        if ch_node:
            m = re.search(r"(\d+)/", ch_node.text)
            if m:
                chapters = int(m.group(1))

        results.append(
            {
                "work_id": extract_work_id_from_blurb(work),
                "hits": get_val("hits"),
                "kudos": get_val("kudos"),
                "comments": get_val("comments"),
                "chapters": chapters,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Cohort scraping
# ---------------------------------------------------------------------------

def scrape_fandom_cohort(
    fandom,
    category,
    warning,
    crossover,
    complete_only,
    timeframe,
    chapter_choice,
    target_chapters,
    total_pages,
    cache,
    start_page=1,
    end_page=None,
):
    """Walks result pages [start_page, end_page] of the filtered search,
    extracting stats purely from search blurbs (no per-work page visits).
    Uses the on-disk page cache to avoid re-requesting pages already
    scraped in a prior run (page 1 is often already cached by
    get_result_page_count), and stops the whole run (rather than trying to
    push through) if AO3 sends repeated hard blocks.

    start_page/end_page let you split one big fandom into separate runs
    (e.g. pages 1-100 today, 200-300 next week) — each run only touches its
    own slice, and results still accumulate into the same CSV/cache/
    distribution files regardless of which pages were covered when.
    """
    fics = []

    end_page = min(end_page, total_pages) if end_page is not None else total_pages
    start_page = max(1, start_page)
    pages_to_fetch = list(range(start_page, end_page + 1))

    print(f"  Scanning pages {start_page}-{end_page} ({len(pages_to_fetch)} of {total_pages} total result page(s)).")

    consecutive_hard_failures = 0

    for i, page in enumerate(pages_to_fetch, start=1):
        url = build_tag_url(fandom, category, warning, crossover, complete_only, timeframe, page)

        cached = cache["pages"].get(url)
        if cached is not None:
            page_results = cached
            print(f"  Page {page}: loaded from cache ({len(page_results)} works), no request made.")
        else:
            polite_delay()
            resp = get_with_retry(url)

            if resp is None or resp.status_code in (403, 503, 525):
                code = resp.status_code if resp is not None else "no response"
                consecutive_hard_failures += 1
                print(
                    f"  Page {page}: HTTP {code} — AO3 is blocking or failing "
                    f"this request ({consecutive_hard_failures}/{MAX_CONSECUTIVE_HARD_FAILURES})."
                )
                if consecutive_hard_failures >= MAX_CONSECUTIVE_HARD_FAILURES:
                    print(
                        "\n  Stopping the scrape: AO3 has blocked several requests in a "
                        "row. This usually means you've been rate-limited. Save what you "
                        "have and wait at least an hour (ideally longer) before running "
                        "again — running again immediately is likely to extend the block, "
                        "not get past it."
                    )
                    break
                continue

            if resp.status_code != 200:
                print(f"  Page {page}: HTTP {resp.status_code}, skipping.")
                continue

            consecutive_hard_failures = 0
            html = resp.text

            page_results = parse_blurb_page(html)
            cache["pages"][url] = page_results
            save_cache(cache)

        for w in page_results:
            chapters = w["chapters"]
            hits = w["hits"]

            if chapter_choice == "1" and chapters != 1:
                continue
            elif chapter_choice == "2" and chapters != target_chapters:
                continue
            elif chapter_choice == "3" and chapters <= 1:
                continue

            if hits >= 50:
                kudos = w["kudos"]
                comments = w["comments"]
                fics.append(
                    {
                        "work_id": w["work_id"],
                        "hits": hits,
                        "kudos": kudos,
                        "comments": comments,
                        "chapters": chapters,
                        "kudos_to_hits": (kudos / hits) * 100,
                        "comments_to_hits": (comments / hits) * 100,
                        "comments_to_kudos": (comments / kudos * 100) if kudos > 0 else 0,
                    }
                )

        if i % 10 == 0 or i == len(pages_to_fetch):
            print(f"  ...page {i}/{len(pages_to_fetch)} done, {len(fics)} qualifying works so far")

    return pd.DataFrame(fics)


# ---------------------------------------------------------------------------
# CSV persistence (append-only, deduped by work_id)
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "run_timestamp", "role", "work_id", "title", "fandom",
    "fandom_scope", "category_filter", "warning_filter", "crossover_filter",
    "chapter_filter", "completion_filter", "timeframe_filter",
    "hits", "kudos", "comments", "chapters",
    "kudos_to_hits", "comments_to_hits", "comments_to_kudos",
]


def load_existing_work_ids(csv_path):
    ids = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("work_id"):
                    ids.add(row["work_id"])
    return ids


def append_results_to_csv(csv_path, rows):
    """Appends rows to csv_path, creating it with a header if it doesn't
    exist yet. Caller is responsible for pre-filtering out duplicates."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Distribution export: lets you scrape a fandom ONCE and then, offline, work
# out where any other fic (given its own hits/kudos/comments) falls on the
# distribution — no need to re-scrape or make extra AO3 requests per fic.
# ---------------------------------------------------------------------------

def make_distribution_key(fandom, filter_meta):
    return "|".join([
        fandom or "ALL_FANDOMS",
        f"cat{filter_meta['category_filter']}",
        f"warn{filter_meta['warning_filter']}",
        f"cross{filter_meta['crossover_filter']}",
        f"chap{filter_meta['chapter_filter']}",
        f"comp{filter_meta['completion_filter']}",
        f"time{filter_meta['timeframe_filter']}",
    ])


def compute_and_save_distribution(csv_path, dist_path, fandom, filter_meta):
    """Recomputes the ratio distributions for this fandom+filter combo from
    the FULL accumulated CSV (not just this run's results), and writes them
    into dist_path keyed by fandom+filters. Because it reads from the whole
    CSV, running this once per fandom per session and letting the CSV grow
    over multiple runs makes each stored distribution more complete over
    time, without needing to re-scrape past pages (they're cached anyway).
    """
    if not os.path.exists(csv_path):
        return None

    df_all = pd.read_csv(csv_path, dtype={
        "category_filter": str, "warning_filter": str, "crossover_filter": str,
        "chapter_filter": str, "completion_filter": str, "timeframe_filter": str,
        "work_id": str,
    })
    if df_all.empty:
        return None

    mask = (
        (df_all["role"] == "cohort")
        & (df_all["fandom"].fillna("") == (fandom or ""))
        & (df_all["category_filter"] == str(filter_meta["category_filter"]))
        & (df_all["warning_filter"] == str(filter_meta["warning_filter"]))
        & (df_all["crossover_filter"] == str(filter_meta["crossover_filter"]))
        & (df_all["chapter_filter"] == str(filter_meta["chapter_filter"]))
        & (df_all["completion_filter"] == str(filter_meta["completion_filter"]))
        & (df_all["timeframe_filter"] == str(filter_meta["timeframe_filter"]))
    )
    subset = df_all[mask]
    if subset.empty:
        return None

    entry = {
        "fandom": fandom or "(all fandoms)",
        **filter_meta,
        "sample_size": int(len(subset)),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "ratios": {},
    }

    percentile_points = [5, 10, 25, 50, 75, 90, 95]
    for col in ["kudos_to_hits", "comments_to_hits", "comments_to_kudos"]:
        values = sorted(float(v) for v in subset[col].tolist())
        entry["ratios"][col] = {
            "values": values,  # full sorted distribution, for manual lookup
            "mean": float(np.mean(values)),
            "percentiles": {
                str(p): float(np.percentile(values, p)) for p in percentile_points
            },
        }

    all_dist = {}
    if os.path.exists(dist_path):
        try:
            with open(dist_path, "r", encoding="utf-8") as f:
                all_dist = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_dist = {}

    key = make_distribution_key(fandom, filter_meta)
    all_dist[key] = entry

    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(all_dist, f, indent=2)

    return entry, key


COOKIE_FILE = "ao3_cookie.txt"


def get_session_cookie_input():
    """Gets the optional session cookie either from ao3_cookie.txt (if
    present) or by prompting interactively.

    Many terminals have a hard length limit on a single pasted line before
    the line-editing buffer stops accepting input (a long-standing quirk on
    macOS in particular) — a 5-cookie header including Cloudflare's
    cf_clearance/cf_bm cookies can easily be 500-1500+ characters and blow
    right past that limit, making Enter appear to do nothing. Reading from
    a file sidesteps this entirely, since text editors don't have that
    restriction.
    """
    if os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            print(f"Found {COOKIE_FILE} — using the cookie from that file (skipping the prompt).")
            return content
        print(f"{COOKIE_FILE} exists but is empty — falling back to the prompt.")

    print("Optional: paste a logged-in AO3 session cookie to include works")
    print("restricted to registered users (copy the 'Cookie' request header")
    print("from your browser's dev tools while logged into AO3).")
    print(
        f"If pasting into this prompt doesn't work (very long cookie strings can "
        f"exceed your terminal's paste limit), instead create a plain text file "
        f"named '{COOKIE_FILE}' in this same folder, paste the cookie into it, "
        f"save it, and re-run the script — it'll be picked up automatically."
    )
    print("Leave blank here to continue anonymously (restricted works excluded).")
    return input("AO3 session cookie (optional): ").strip()


def main():
    cookie_in = get_session_cookie_input()
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

    print("\nCategory: [0] All Categories, [1] M/M, [2] F/F, [3] F/M, [4] Multi, [5] Gen")
    cat_in = input("Select Category (0-5): ").strip()

    print(
        "\nArchive Warnings: [0] All Warnings, [1] No Warnings Apply, "
        "[2] Creator Chose Not To Use, [3] Violence, [4] Major Character Death"
    )
    warn_in = input("Select Warning (0-4): ").strip()

    print("\nCrossovers: [0] All Works, [1] No Crossovers (Single Fandom), [2] Crossovers Only")
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

    print("\nChecking how many result pages match these filters...")
    cache = load_cache()
    total_pages, first_page_url = get_result_page_count(
        fandom_query, cat_in, warn_in, cross_in, comp_in, time_in, cache
    )

    if total_pages == 0:
        print("No matching works found for these filters.")
        return

    est_seconds_full = total_pages * 7.5  # midpoint of the 5-10s jitter range
    print(
        f"Found {total_pages} result page(s) (~{total_pages * 20} works). "
        f"Scanning all of them would take roughly "
        f"{int(est_seconds_full // 60)}m at AO3's rate limit."
    )
    print(
        "You can scan the whole thing, or just a slice of pages (e.g. pages "
        "1-100 now, 200-300 in a later run). Cached/already-CSV'd pages are "
        "always skipped automatically, so slices can overlap safely."
    )
    start_in = input(f"Start page (1-{total_pages}, blank = 1): ").strip()
    end_in = input(f"End page (1-{total_pages}, blank = {total_pages}): ").strip()
    start_page = int(start_in) if start_in.isdigit() else 1
    end_page = int(end_in) if end_in.isdigit() else total_pages

    print(
        "\nCollecting cohort comparison data from AO3 "
        "(scraping search result cards only, no per-work page visits)..."
    )
    df = scrape_fandom_cohort(
        fandom_query, cat_in, warn_in, cross_in, comp_in, time_in,
        chap_in, target["chapters"],
        total_pages=total_pages,
        cache=cache,
        start_page=start_page,
        end_page=end_page,
    )

    # --- Persist results to the running CSV, deduped by work_id ---
    existing_ids = load_existing_work_ids(CSV_FILE)
    run_ts = datetime.now().isoformat(timespec="seconds")
    filter_meta = {
        "fandom_scope": "same" if fandom_choice == "0" else "all",
        "category_filter": cat_in,
        "warning_filter": warn_in,
        "crossover_filter": cross_in,
        "chapter_filter": chap_in,
        "completion_filter": comp_in,
        "timeframe_filter": time_in,
    }

    new_rows = []
    if target.get("work_id") and target["work_id"] not in existing_ids:
        new_rows.append({
            "run_timestamp": run_ts, "role": "target", "work_id": target["work_id"],
            "title": target["title"], "fandom": target["fandom"], **filter_meta,
            "hits": target["hits"], "kudos": target["kudos"], "comments": target["comments"],
            "chapters": target["chapters"], "kudos_to_hits": target["kudos_to_hits"],
            "comments_to_hits": target["comments_to_hits"], "comments_to_kudos": target["comments_to_kudos"],
        })

    skipped_dupes = 0
    for _, r in df.iterrows():
        wid = r.get("work_id")
        if wid and wid in existing_ids:
            skipped_dupes += 1
            continue
        if wid:
            existing_ids.add(wid)
        new_rows.append({
            "run_timestamp": run_ts, "role": "cohort", "work_id": wid,
            "title": "", "fandom": fandom_query, **filter_meta,
            "hits": r["hits"], "kudos": r["kudos"], "comments": r["comments"],
            "chapters": r["chapters"], "kudos_to_hits": r["kudos_to_hits"],
            "comments_to_hits": r["comments_to_hits"], "comments_to_kudos": r["comments_to_kudos"],
        })

    if new_rows:
        append_results_to_csv(CSV_FILE, new_rows)
    print(
        f"\nAppended {len(new_rows)} new row(s) to {CSV_FILE} "
        f"({skipped_dupes} already-seen work(s) skipped)."
    )

    dist_result = compute_and_save_distribution(CSV_FILE, DIST_FILE, fandom_query, filter_meta)
    if dist_result:
        dist_entry, dist_key = dist_result
        print(
            f"Saved distribution ({dist_entry['sample_size']} works) to "
            f"{DIST_FILE} under key:\n  {dist_key}"
        )
        print(
            "This file holds the full sorted list of ratios plus percentile "
            "breakpoints for this fandom+filter combo — you can look up where "
            "any other fic's ratios fall against it offline, without scraping again."
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
    print(f"  Performance:          Outperforms {k_perc:.1f}% of cohort fics (Top {100 - k_perc:.1f}%)")

    print("\nComments:Hits Ratio Comparison:")
    print(f"  Target Fic Ratio:     {target['comments_to_hits']:.2f}%")
    print(f"  Cohort Average Ratio: {avg_c_ratio:.2f}%")
    print(f"  Performance:          Outperforms {c_perc:.1f}% of cohort fics (Top {100 - c_perc:.1f}%)")


if __name__ == "__main__":
    main()