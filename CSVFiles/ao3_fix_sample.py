
#!/usr/bin/env python3
"""
ao3_sample_fixer.py
===================
Reads checker_results.json (produced by ao3_sample_checker.py), identifies
the TWO worst deviating dimensions per fandom, then randomly swaps works
to bring the sample back into proportion.

Randomness strategy (for AO3's 5,000 page / 100K work limit):
  1. RANDOM DATE FILTER: pick a random date in the fandom's lifespan,
     search AO3 for fandom + required tag + that single date. Daily
     results are small, so ALL works for that day are accessible.
  2. RANDOM SORT + PAGE: randomly pick a sort column (hits, kudos,
     word count, date, etc.) AND direction, then a random page.
     Different sorts surface different 100K work windows.
  3. CSV POOL FALLBACK: if AO3 fails, pull from ao3_results.csv /
     ao3_results_hp.csv which already have large scraped pools.

Usage:
    python3 ao3_sample_fixer.py
    python3 ao3_sample_fixer.py --dry-run
    python3 ao3_sample_fixer.py --csv ao3_testing_sample.csv --cookie-file ao3_cookie.txt
    python3 ao3_sample_fixer.py --pool-csv ao3_results.csv ao3_results_hp.csv
    python3 ao3_sample_fixer.py --threshold 3.0 --max-swaps-per-dim 10
"""

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from urllib.parse import quote, urlencode

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fix AO3 sample representativeness.")
parser.add_argument("--csv", default="ao3_testing_sample.csv",
                    help="Sample CSV to fix (modified in place, backup saved as .bak).")
parser.add_argument("--cookie-file", default="ao3_cookie.txt",
                    help="AO3 cookie file for authenticated access.")
parser.add_argument("--sep", default=" | ",
                    help="Tag separator in CSV.")
parser.add_argument("--threshold", type=float, default=5.0,
                    help="Only fix dimensions whose worst deviation exceeds this (pp).")
parser.add_argument("--max-swaps-per-dim", type=int, default=15,
                    help="Maximum works to swap per dimension (prevents over correction).")
parser.add_argument("--dry-run", action="store_true",
                    help="Show planned swaps without making any changes.")
parser.add_argument("--pool-csv", nargs="*", default=None,
                    help="CSV file(s) to use as fallback random pool.")

ARGS = parser.parse_args()
CSV_PATH = os.path.abspath(ARGS.csv)
TAG_SEP = ARGS.sep
THRESHOLD = ARGS.threshold


# ══════════════════════════════════════════════════════════════
# AO3 CONSTANTS
# ══════════════════════════════════════════════════════════════

RATING_IDS = {
    "Not Rated": 9, "General Audiences": 10,
    "Teen And Up Audiences": 11, "Mature": 12, "Explicit": 13,
}
CATEGORY_IDS = {
    "F/F": 116, "F/M": 22, "Gen": 21,
    "M/M": 23, "Multi": 2246, "Other": 24,
}
WARNING_IDS = {
    "No Archive Warnings Apply": 14,
    "Creator Chose Not To Use Archive Warnings": 16,
    "Graphic Depictions Of Violence": 17,
    "Major Character Death": 18,
    "Underage Sex": 19,
    "Rape/Non-Con": 20,
}

# Plain text fandom names for /works/search (NOT canonical tag slugs).
# /works/search does a text match on fandom_names, so these don't need
# to be exact canonical tag names. Much more forgiving than /tags/...
FANDOM_SEARCH_NAMES = {
    "Harry Potter": "Harry Potter - J. K. Rowling",
    "Arcane":       "Arcane: League of Legends",
    "Avatar":       "Avatar: The Last Airbender",
}

# Date ranges per fandom (for random date strategy)
FANDOM_DATE_RANGES = {
    "Harry Potter": (date(2009, 1, 1), date(2026, 10, 20)),
    "Arcane":       (date(2021, 11, 1), date(2026, 10, 20)),
    "Avatar":       (date(2009, 1, 1), date(2026, 10, 20)),
}

SORT_COLUMNS = [
    "revised_at", "created_at", "word_count",
    "hits", "kudos_count", "comments_count", "bookmarks_count",
]

# Browser config
COOKIE_DOMAIN = ".archiveofourown.org"
DEFAULT_COOKIES = [
    {"name": "view_adult", "value": "true", "domain": COOKIE_DOMAIN, "path": "/"},
]
HEADLESS_UA_TOKEN = "HeadlessChrome"
HEADLESS = os.environ.get("AO3_HEADLESS", "").strip().lower() in {"1", "true", "yes"}
BROWSER_PROFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ao3_browser_profile_fixer"
)
REQUEST_GAP = (10.0, 15.0)
CHALLENGE_WAIT = 30

CSV_FIELDS = [
    "sample_id", "work_id", "url", "title", "fandom",
    "rating", "warnings", "category",
    "relationships", "characters", "freeform_tags",
    "date_published", "date_updated",
    "words", "chapters", "language",
    "hits", "kudos", "bookmarks", "comments",
    "kudos_to_hits", "bookmarks_to_hits", "comments_to_hits",
    "comments_to_kudos",
    "kudos_per_10k_words", "bookmarks_per_10k_words", "comments_per_10k_words",
    "tags_enriched",
]

FANDOM_KEYWORDS = {
    "Harry Potter": ["harry potter", "j. k. rowling"],
    "Arcane":       ["arcane", "league of legends"],
    "Avatar":       ["avatar", "last airbender", "atla"],
}


# ══════════════════════════════════════════════════════════════
# CSV I/O
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


def save_csv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore",
            restval="", quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in rows:
            safe = {field: row.get(field, "") or "" for field in CSV_FIELDS}
            writer.writerow(safe)
        f.flush()
        os.fsync(f.fileno())
    backup = path + ".bak"
    if os.path.exists(path):
        try:
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
        except OSError:
            pass
    os.rename(tmp, path)


def detect_fandom(row):
    fandom_text = row.get("fandom", "").lower()
    for name, keywords in FANDOM_KEYWORDS.items():
        for kw in keywords:
            if kw in fandom_text:
                return name
    return "Unknown"


# ══════════════════════════════════════════════════════════════
# BROWSER SESSION
# ══════════════════════════════════════════════════════════════

class BrowserSession:
    def __init__(self):
        self._pw = None
        self._ctx = None
        self._page = None
        self._cookies = list(DEFAULT_COOKIES)
        self._last_req = None

    def start(self):
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._ctx, self._page = self._launch()
        self._ctx.set_default_navigation_timeout(60_000)
        self._ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )
        if self._cookies:
            self._ctx.add_cookies(self._cookies)
        self._warm_up()

    def _launch(self):
        ctx = self._launch_ctx()
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        from playwright.sync_api import Error as PwError
        try:
            ua = page.evaluate("navigator.userAgent")
        except PwError:
            return ctx, page
        if ua and HEADLESS_UA_TOKEN in ua:
            ctx.close()
            ctx = self._launch_ctx(
                user_agent=ua.replace(HEADLESS_UA_TOKEN, "Chrome")
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
        return ctx, page

    def _launch_ctx(self, user_agent=None):
        from playwright.sync_api import Error as PwError
        opts = {
            "headless": HEADLESS,
            "viewport": {"width": 1280, "height": 900},
            "locale": "en-US",
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if user_agent:
            opts["user_agent"] = user_agent
        try:
            return self._pw.chromium.launch_persistent_context(
                os.path.join(BROWSER_PROFILE, "chrome"),
                channel="chrome", **opts,
            )
        except PwError:
            return self._pw.chromium.launch_persistent_context(
                os.path.join(BROWSER_PROFILE, "chromium"), **opts,
            )

    def _throttle(self):
        gap = random.uniform(*REQUEST_GAP)
        if self._last_req is not None:
            remaining = gap - (time.monotonic() - self._last_req)
            if remaining > 0:
                time.sleep(remaining)
        self._last_req = time.monotonic()

    def _warm_up(self):
        from playwright.sync_api import Error as PwError, TimeoutError as PwTimeout
        try:
            self._throttle()
            self._page.goto(
                "https://archiveofourown.org/", wait_until="domcontentloaded"
            )
            self._settle()
        except (PwError, PwTimeout):
            pass

    def _settle(self):
        from playwright.sync_api import Error as PwError
        try:
            body = self._page.content()
        except PwError:
            time.sleep(2)
            body = self._page.content()
        if not _is_challenge(body):
            return body
        print("    Interstitial detected, waiting...")
        deadline = time.monotonic() + CHALLENGE_WAIT
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                body = self._page.content()
            except PwError:
                continue
            if not _is_challenge(body):
                print("    Cleared.")
                return body
        return body

    def add_cookies(self, cookie_dict):
        cookies = [
            {"name": n, "value": v, "domain": COOKIE_DOMAIN, "path": "/"}
            for n, v in cookie_dict.items()
        ]
        self._cookies.extend(cookies)
        if self._ctx:
            self._ctx.add_cookies(cookies)

    def get(self, url, timeout=60):
        from playwright.sync_api import Error as PwError, TimeoutError as PwTimeout
        self.start()
        self._throttle()
        resp = self._page.goto(
            url, wait_until="domcontentloaded", timeout=int(timeout * 1000)
        )
        status = resp.status if resp else 0
        headers = {}
        try:
            headers = resp.all_headers() if resp else {}
        except PwError:
            pass
        body = self._settle()
        self._last_req = time.monotonic()
        if status == 200 and _is_challenge(body):
            status = 403
        return _Resp(status, body, headers, self._page.url)

    def close(self):
        if self._ctx:
            try:
                self._ctx.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._pw = self._ctx = self._page = None


class _Resp:
    def __init__(self, status_code, text, headers, url):
        self.status_code = status_code
        self.text = text
        self.headers = headers
        self.url = url


def _is_challenge(body):
    if not body:
        return False
    low = body.lower()
    if 'id="header"' in low or "/users/logout" in low:
        return False
    return any(m in low for m in (
        "cf-browser-verification", "cf_chl_opt",
        "challenge-platform", "__cf_chl",
    ))


def _get_with_retry(browser, url, max_retries=3):
    from playwright.sync_api import Error as PwError, TimeoutError as PwTimeout
    for attempt in range(max_retries):
        try:
            resp = browser.get(url, timeout=60)
        except (PwError, PwTimeout):
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        if resp.status_code in (520, 521, 522, 523, 524, 525, 526, 527, 0):
            time.sleep(20 * (attempt + 1))
            continue
        return resp
    return None


def _parse_cookie_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    raw = re.sub(r"^\s*cookie\s*:\s*", "", raw, flags=re.IGNORECASE)
    cookies = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,60}", k):
            cookies[k] = v.strip().strip("\"'")
    return cookies


# ══════════════════════════════════════════════════════════════
# AO3 SEARCH & SCRAPE
# ══════════════════════════════════════════════════════════════

def build_search_url(fandom_name, dim_field, tag_name,
                     sort_col="revised_at", sort_dir="desc",
                     page=1, date_from=None, date_to=None):
    """Build an AO3 /works/search URL (same endpoint the scraper uses).

    Uses /works/search with work_search[fandom_names] which does a
    text match. This is forgiving with fandom names unlike /tags/...
    which requires the exact canonical tag slug and 404s otherwise.
    """
    base = "https://archiveofourown.org/works/search?"

    fandom_text = FANDOM_SEARCH_NAMES.get(fandom_name, fandom_name)

    params = [
        ("utf8", "\u2713"),
        ("commit", "Sort and Filter"),
        ("page", str(page)),
        ("work_search[sort_column]", sort_col),
        ("work_search[sort_direction]", sort_dir),
        ("work_search[fandom_names]", fandom_text),
    ]

    # Dimension specific filter
    if dim_field == "rating" and tag_name in RATING_IDS:
        params.append(
            ("work_search[rating_ids][]", str(RATING_IDS[tag_name]))
        )
    elif dim_field == "category" and tag_name in CATEGORY_IDS:
        params.append(
            ("work_search[category_ids][]", str(CATEGORY_IDS[tag_name]))
        )
    elif dim_field == "warnings" and tag_name in WARNING_IDS:
        params.append(
            ("work_search[archive_warning_ids][]", str(WARNING_IDS[tag_name]))
        )
    elif dim_field == "relationships":
        params.append(("work_search[relationship_names]", tag_name))
    elif dim_field == "characters":
        params.append(("work_search[character_names]", tag_name))

    if date_from:
        params.append(("work_search[date_from]", date_from))
    if date_to:
        params.append(("work_search[date_to]", date_to))

    return base + urlencode(params)


def parse_search_page(html):
    """Parse an AO3 search results page.

    Handles both:
      /works/search   -> h3.heading "1 - 20 of 301,366 Works found"
      /tags/.../works -> h2.heading "301,366 Works in ..."
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    total = 0
    for tag in ["h3", "h2"]:
        for heading in soup.find_all(tag, class_="heading"):
            text = heading.get_text()
            m = re.search(r"([\d,]+)\s+Works?\b", text)
            if m:
                total = int(m.group(1).replace(",", ""))
                break
        if total > 0:
            break

    work_ids = []
    for li in soup.select("li.work.blurb"):
        wid = li.get("id", "")
        m = re.search(r"work_(\d+)", wid)
        if m:
            work_ids.append(m.group(1))

    return total, work_ids


def fetch_random_work_id_ao3(browser, fandom_name, dim_field, tag_name,
                              exclude_ids, max_attempts=8):
    """Find a truly random work ID on AO3 using /works/search.

    Attempts 0..3: random single-date filter (reaches ANY work ever posted)
    Attempts 4..7: random sort column + random page (broad coverage)
    """
    date_start, date_end = FANDOM_DATE_RANGES.get(
        fandom_name, (date(2009, 1, 1), date(2026, 10, 20))
    )
    total_days = (date_end - date_start).days

    for attempt in range(max_attempts):

        # ── Strategy 1: random date (attempts 0..3) ──────────
        if attempt < 4:
            rand_date = date_start + timedelta(
                days=random.randint(0, total_days)
            )
            ds = rand_date.strftime("%Y-%m-%d")
            sort_col = random.choice(SORT_COLUMNS)

            url = build_search_url(
                fandom_name, dim_field, tag_name,
                sort_col=sort_col, page=1,
                date_from=ds, date_to=ds,
            )
            print(f"    [attempt {attempt+1}] random date {ds}, "
                  f"sort={sort_col}")

            resp = _get_with_retry(browser, url)
            if not resp or resp.status_code != 200:
                print(f"    HTTP {resp.status_code if resp else 'none'}")
                continue

            total, work_ids = parse_search_page(resp.text)
            if total == 0 or not work_ids:
                print(f"    No works on {ds}, trying another date...")
                continue

            # If multi-page results for that day, pick a random page
            if total > 20:
                max_pg = min(math.ceil(total / 20), 5000)
                rand_pg = random.randint(1, max_pg)
                if rand_pg != 1:
                    url2 = build_search_url(
                        fandom_name, dim_field, tag_name,
                        sort_col=sort_col, page=rand_pg,
                        date_from=ds, date_to=ds,
                    )
                    resp2 = _get_with_retry(browser, url2)
                    if resp2 and resp2.status_code == 200:
                        _, work_ids = parse_search_page(resp2.text)

        # ── Strategy 2: random sort + page (attempts 4..7) ──
        else:
            sort_col = random.choice(SORT_COLUMNS)
            sort_dir = random.choice(["asc", "desc"])

            # First get total count
            url0 = build_search_url(
                fandom_name, dim_field, tag_name,
                sort_col=sort_col, sort_dir=sort_dir, page=1,
            )
            print(f"    [attempt {attempt+1}] sort={sort_col} {sort_dir}, "
                  f"getting count...")
            resp0 = _get_with_retry(browser, url0)
            if not resp0 or resp0.status_code != 200:
                print(f"    Search failed")
                continue

            total, _ = parse_search_page(resp0.text)
            if total == 0:
                print(f"    No results")
                continue

            max_page = min(math.ceil(total / 20), 5000)
            rand_page = random.randint(1, max_page)

            url = build_search_url(
                fandom_name, dim_field, tag_name,
                sort_col=sort_col, sort_dir=sort_dir, page=rand_page,
            )
            print(f"    page {rand_page}/{max_page} ({total:,} total)")

            resp = _get_with_retry(browser, url)
            if not resp or resp.status_code != 200:
                continue
            _, work_ids = parse_search_page(resp.text)

        # ── Pick a random work from the page ─────────────────
        candidates = [wid for wid in work_ids if wid not in exclude_ids]
        if not candidates:
            print(f"    No new candidates on this page")
            continue

        chosen = random.choice(candidates)
        print(f"    Found candidate: work {chosen}")
        return chosen

    return None


def scrape_work(browser, work_id):
    """Visit a single AO3 work page and extract ALL metadata."""
    from bs4 import BeautifulSoup

    url = f"https://archiveofourown.org/works/{work_id}"
    resp = _get_with_retry(browser, url)
    if not resp or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
        return None

    def _tags(dd_class):
        node = soup.find("dd", class_=dd_class)
        if node is None:
            return ""
        links = [a.get_text(" ", strip=True) for a in node.select("li a")]
        return TAG_SEP.join(links) if links else node.get_text(" ", strip=True)

    def _stat(class_name):
        node = stats_dl.find("dd", class_=class_name)
        if node is None:
            return 0
        m = re.search(r"\d+", node.get_text(strip=True).replace(",", ""))
        return int(m.group()) if m else 0

    r = {}
    r["work_id"] = work_id
    r["url"] = url

    title_node = soup.find("h2", class_="title")
    r["title"] = title_node.get_text(" ", strip=True) if title_node else ""

    fandom_links = soup.select("dd.fandom.tags ul.commas li a")
    r["fandom"] = (
        fandom_links[0].get_text(" ", strip=True) if fandom_links else ""
    )

    r["rating"]        = _tags("rating")
    r["warnings"]      = _tags("warning")
    r["category"]      = _tags("category")
    r["relationships"] = _tags("relationship")
    r["characters"]    = _tags("character")
    r["freeform_tags"] = _tags("freeform")

    pub = stats_dl.find("dd", class_="published")
    upd = stats_dl.find("dd", class_="status")
    r["date_published"] = pub.get_text(" ", strip=True) if pub else ""
    r["date_updated"] = (
        upd.get_text(" ", strip=True) if upd else r["date_published"]
    )

    r["hits"]      = _stat("hits")
    r["kudos"]     = _stat("kudos")
    r["bookmarks"] = _stat("bookmarks")
    r["comments"]  = _stat("comments")
    r["words"]     = _stat("words")

    ch = stats_dl.find("dd", class_="chapters")
    if ch:
        m2 = re.search(r"(\d+)\s*/", ch.get_text(" ", strip=True))
        r["chapters"] = int(m2.group(1)) if m2 else 1
    else:
        r["chapters"] = 1

    lang = soup.find("dd", class_="language")
    r["language"] = lang.get_text(" ", strip=True) if lang else ""

    # Engagement ratios
    h, k, b, c, w = (
        r["hits"], r["kudos"], r["bookmarks"], r["comments"], r["words"]
    )
    r["kudos_to_hits"]           = round(k / h * 100, 4) if h > 0 else 0
    r["bookmarks_to_hits"]       = round(b / h * 100, 4) if h > 0 else 0
    r["comments_to_hits"]        = round(c / h * 100, 4) if h > 0 else 0
    r["comments_to_kudos"]       = round(c / k * 100, 4) if k > 0 else 0
    r["kudos_per_10k_words"]     = round(k / w * 10000, 4) if w > 0 else 0
    r["bookmarks_per_10k_words"] = round(b / w * 10000, 4) if w > 0 else 0
    r["comments_per_10k_words"]  = round(c / w * 10000, 4) if w > 0 else 0
    r["tags_enriched"] = "yes"

    return r


# ══════════════════════════════════════════════════════════════
# CSV POOL FALLBACK
# ══════════════════════════════════════════════════════════════

def load_csv_pool(fandom_name, pool_paths=None):
    """Load large scraped CSVs as a random pool for a fandom."""
    search_files = pool_paths or []
    if not search_files:
        candidates = [
            "ao3_results.csv", "ao3_results_hp.csv",
            "ao3_results_arcane.csv", "ao3_results_avatar.csv",
        ]
        search_files = [f for f in candidates if os.path.exists(f)]

    if not search_files:
        return []

    keywords = FANDOM_KEYWORDS.get(fandom_name, [])
    if not keywords:
        return []

    pool = []
    for fp in search_files:
        try:
            rows = read_csv(fp)
            for row in rows:
                fd = row.get("fandom", "").lower()
                if any(kw in fd for kw in keywords):
                    pool.append(row)
        except Exception as e:
            print(f"    Warning: could not read {fp}: {e}")
    return pool


def pick_from_csv_pool(pool, dim_field, tag_name, exclude_ids):
    """Pick a random work from the CSV pool that has the required tag."""
    def has_tag(row, field, tag):
        val = row.get(field, "").lower()
        tag_l = tag.lower()
        for sep in [TAG_SEP, ",", "|"]:
            for p in val.split(sep):
                p = p.strip()
                if tag_l == p or tag_l in p or p in tag_l:
                    return True
        return False

    candidates = [
        row for row in pool
        if row.get("work_id", "") not in exclude_ids
        and has_tag(row, dim_field, tag_name)
    ]

    if not candidates:
        return None
    return random.choice(candidates)


# ══════════════════════════════════════════════════════════════
# CHECKER INTEGRATION
# ══════════════════════════════════════════════════════════════

def run_checker_and_load(csv_path):
    """Run ao3_sample_checker.py as subprocess, read its JSON output."""
    checker = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ao3_sample_checker.py",
    )
    if not os.path.exists(checker):
        sys.exit(f"ERROR: Cannot find {checker}")

    print(f"Running sample checker on {os.path.basename(csv_path)}...")
    result = subprocess.run(
        [sys.executable, checker, "--csv", csv_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Checker stderr:\n{result.stderr}")
        sys.exit("Checker failed.")

    json_path = os.path.join(
        os.path.dirname(checker), "checker_results.json"
    )
    if not os.path.exists(json_path):
        json_path = "checker_results.json"
    if not os.path.exists(json_path):
        sys.exit("ERROR: checker_results.json not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# SWAP LOGIC
# ══════════════════════════════════════════════════════════════

def identify_problems(checker_results, threshold):
    """Find the 2 worst dimensions per fandom exceeding threshold."""
    problems = []

    for fandom_name, fdata in checker_results.items():
        dims = fdata.get("dimensions", {})
        sorted_dims = sorted(
            dims.items(), key=lambda x: x[1]["worst_diff"], reverse=True
        )

        picked = 0
        for dim_name, dim_data in sorted_dims:
            if picked >= 2:
                break
            if dim_data["worst_diff"] <= threshold:
                continue

            tags = dim_data["tags"]
            under_tag, under_info = min(
                tags.items(), key=lambda x: x[1]["diff"]
            )
            over_tag, over_info = max(
                tags.items(), key=lambda x: x[1]["diff"]
            )

            if under_info["diff"] > -threshold:
                continue

            problems.append({
                "fandom": fandom_name,
                "dimension": dim_name,
                "field": dim_data["field"],
                "is_single_value": dim_data["is_single_value"],
                "worst_diff": dim_data["worst_diff"],
                "under_tag": under_tag,
                "under_diff": under_info["diff"],
                "under_count": under_info["count"],
                "under_pop_pct": under_info["pop_pct"],
                "over_tag": over_tag,
                "over_diff": over_info["diff"],
                "over_count": over_info["count"],
                "over_pop_pct": over_info["pop_pct"],
                "denominator": under_info["denominator"],
                "n_works": fdata["n_works"],
            })
            picked += 1

    problems.sort(key=lambda x: x["worst_diff"], reverse=True)
    return problems


def calculate_swap_count(problem, max_swaps):
    """How many works to swap: enough to close the gap, capped safely."""
    n = problem["n_works"]
    denom = problem["denominator"]
    is_single = problem["is_single_value"]

    # How many under-tag works needed to reach population %?
    target_pct = problem["under_pop_pct"] / 100
    current_under = problem["under_count"]
    if is_single:
        target_count = round(n * target_pct)
    else:
        target_count = round(denom * target_pct)
    needed = max(0, target_count - current_under)

    # How many over-tag works can we safely remove?
    over_target_pct = problem["over_pop_pct"] / 100
    current_over = problem["over_count"]
    if is_single:
        over_target = round(n * over_target_pct)
    else:
        over_target = round(denom * over_target_pct)
    removable = max(0, current_over - over_target)

    swaps = min(needed, removable, max_swaps)
    if swaps == 0 and needed > 0 and removable > 0:
        swaps = 1
    return swaps


def find_removable_indices(rows, fandom_name, field, over_tag,
                            under_tag, n_remove):
    """Find row indices to remove: works with over_tag, preferring
    those that do NOT also have under_tag. Randomized."""
    def tag_in_field(row, fld, tag):
        val = row.get(fld, "").lower()
        tag_l = tag.lower()
        if val == tag_l:
            return True
        for part in val.split(TAG_SEP):
            p = part.strip()
            if tag_l == p or tag_l in p or p in tag_l:
                return True
        return False

    safe = []    # has over tag, NOT under tag (safest to remove)
    risky = []   # has BOTH tags

    for i, row in enumerate(rows):
        if detect_fandom(row) != fandom_name:
            continue
        if not tag_in_field(row, field, over_tag):
            continue
        if tag_in_field(row, field, under_tag):
            risky.append(i)
        else:
            safe.append(i)

    random.shuffle(safe)
    random.shuffle(risky)
    ordered = safe + risky
    return ordered[:n_remove]


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    csv_path = CSV_PATH
    dry_run = ARGS.dry_run
    max_swaps = ARGS.max_swaps_per_dim

    print(f"{'=' * 60}")
    print("AO3 SAMPLE FIXER")
    print(f"{'=' * 60}")
    print(f"  CSV:           {csv_path}")
    print(f"  Threshold:     {THRESHOLD} pp")
    print(f"  Max swaps/dim: {max_swaps}")
    print(f"  Dry run:       {dry_run}")
    print()

    # ── 1. Run checker ───────────────────────────────────────
    checker_results = run_checker_and_load(csv_path)
    print()

    # ── 2. Identify the 2 worst dimensions per fandom ────────
    problems = identify_problems(checker_results, THRESHOLD)

    if not problems:
        print("All dimensions are within threshold. "
              "Sample looks representative!")
        return

    print(f"{'=' * 60}")
    print(f"PROBLEMS FOUND: {len(problems)}")
    print(f"{'=' * 60}")

    for i, p in enumerate(problems, 1):
        swaps = calculate_swap_count(p, max_swaps)
        print(f"\n  {i}. {p['fandom']} / {p['dimension'].upper()} "
              f"(worst: {p['worst_diff']:.1f} pp)")
        print(f"     Under represented: {p['under_tag']} "
              f"({p['under_diff']:+.1f} pp, n={p['under_count']})")
        print(f"     Over represented:  {p['over_tag']} "
              f"({p['over_diff']:+.1f} pp, n={p['over_count']})")
        print(f"     Planned swaps: {swaps}")

    if dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN: no changes will be made.")
        print(f"{'=' * 60}")
        return

    # ── 3. Load sample ───────────────────────────────────────
    rows = read_csv(csv_path)
    existing_ids = {row.get("work_id", "") for row in rows}
    next_sample_id = max(
        (int(row.get("sample_id", 0) or 0) for row in rows), default=0
    ) + 1

    # ── 4. Start browser ────────────────────────────────────
    print("\nStarting browser for AO3 access...")
    browser = BrowserSession()
    cookies = _parse_cookie_file(ARGS.cookie_file)
    if cookies:
        browser.add_cookies(cookies)
        print(f"  Loaded cookies: {', '.join(sorted(cookies.keys()))}")
    browser.start()

    csv_pool_cache = {}
    total_swaps_done = 0

    try:
        for prob in problems:
            fandom = prob["fandom"]
            dim = prob["dimension"]
            field = prob["field"]
            under_tag = prob["under_tag"]
            over_tag = prob["over_tag"]
            n_swaps = calculate_swap_count(prob, max_swaps)

            if n_swaps == 0:
                print(f"\nSkipping {fandom}/{dim}: 0 swaps calculated")
                continue

            print(f"\n{'=' * 60}")
            print(f"FIXING: {fandom} / {dim.upper()}")
            print(f"  Remove up to {n_swaps} '{over_tag}' works")
            print(f"  Add {n_swaps} '{under_tag}' works")
            print(f"{'=' * 60}")

            # Find works to remove
            remove_indices = find_removable_indices(
                rows, fandom, field, over_tag, under_tag, n_swaps
            )
            if len(remove_indices) < n_swaps:
                print(f"  Only {len(remove_indices)} removable works "
                      f"(wanted {n_swaps})")
                n_swaps = len(remove_indices)

            if n_swaps == 0:
                print("  No works to remove. Skipping.")
                continue

            # ── Fetch random replacements ────────────────────
            replacements = []
            for s in range(n_swaps):
                print(f"\n  Swap {s+1}/{n_swaps}: "
                      f"finding '{under_tag}' work...")

                # Try AO3 first
                work_id = fetch_random_work_id_ao3(
                    browser, fandom, field, under_tag, existing_ids
                )

                if work_id:
                    print(f"    Scraping work {work_id}...")
                    work_data = scrape_work(browser, work_id)
                    if work_data:
                        work_data["sample_id"] = str(next_sample_id)
                        next_sample_id += 1
                        replacements.append(work_data)
                        existing_ids.add(work_id)
                        title = work_data.get("title", "?")[:55]
                        print(f"    OK: {title}")
                        continue
                    else:
                        print(f"    Scrape failed for {work_id}")

                # Fallback: CSV pool
                print(f"    Trying CSV pool fallback...")
                if fandom not in csv_pool_cache:
                    csv_pool_cache[fandom] = load_csv_pool(
                        fandom, ARGS.pool_csv
                    )
                    n_pool = len(csv_pool_cache[fandom])
                    print(f"    Pool: {n_pool} works for {fandom}")

                pool_pick = pick_from_csv_pool(
                    csv_pool_cache[fandom], field,
                    under_tag, existing_ids,
                )
                if pool_pick:
                    pool_pick["sample_id"] = str(next_sample_id)
                    next_sample_id += 1
                    pool_pick["tags_enriched"] = "no"
                    replacements.append(pool_pick)
                    wid = pool_pick.get("work_id", "")
                    existing_ids.add(wid)
                    title = pool_pick.get("title", "?")[:55]
                    print(f"    CSV pool: {title}")
                else:
                    print(f"    No candidates found. Skipping swap.")

            if not replacements:
                print(f"  No replacements found for {fandom}/{dim}.")
                continue

            # ── Execute the swaps ────────────────────────────
            # Remove (high index first to preserve earlier indices)
            to_remove = sorted(
                remove_indices[:len(replacements)], reverse=True
            )

            print(f"\n  REMOVING {len(to_remove)} works:")
            for idx in to_remove:
                r = rows[idx]
                sid = r.get("sample_id", "?")
                title = r.get("title", "?")[:45]
                tag_val = r.get(field, "?")[:35]
                print(f"    #{sid} {title}  [{tag_val}]")
                rows.pop(idx)

            print(f"  ADDING {len(replacements)} works:")
            for rep in replacements:
                sid = rep.get("sample_id", "?")
                title = rep.get("title", "?")[:45]
                tag_val = rep.get(field, "?")[:35]
                print(f"    #{sid} {title}  [{tag_val}]")
                rows.append(rep)

            total_swaps_done += len(replacements)

            # Save after each dimension fix
            save_csv(csv_path, rows)
            print(f"  Saved. Sample now has {len(rows)} works.")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        save_csv(csv_path, rows)
        print("Progress saved.")
    finally:
        browser.close()

    # ── 5. Re-run checker to verify ──────────────────────────
    if total_swaps_done > 0:
        print(f"\n{'=' * 60}")
        print(f"VERIFICATION: re-running checker after "
              f"{total_swaps_done} swaps")
        print(f"{'=' * 60}")
        run_checker_and_load(csv_path)

        print(f"\nDone! {total_swaps_done} swaps completed.")
        print(f"  Updated CSV:  {csv_path}")
        print(f"  Backup:       {csv_path}.bak")
        print(f"  Summary:      checker_summary.txt")
        print(f"  JSON:         checker_results.json")
        print()
        print("If deviations are still above threshold, run again.")
        print("Each run is random so repeated runs converge.")
    else:
        print("\nNo swaps were made.")


if __name__ == "__main__":
    main()




