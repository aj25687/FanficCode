
#!/usr/bin/env python3
"""
ao3_enrich_sample.py
====================
Reads ao3_testing_sample.csv (which has URLs already filled in),
visits each work's individual AO3 page, and populates every column
from the full work metadata.

THE COMMA FIX:
  Multi-value tag fields (relationships, characters, freeform_tags,
  warnings, category) use " | " (pipe) as the separator instead of
  commas.  This means CSV commas never collide with tag separators
  and the file stays perfectly parseable.

  Example:  Draco Malfoy/Harry Potter | Hermione Granger/Ron Weasley

RESUME SUPPORT:
  Saves after every single work. If interrupted, re-run and it picks
  up where it left off (checks whether tags_enriched == "yes").

Usage:
    python3 ao3_enrich_sample.py
    python3 ao3_enrich_sample.py --csv path/to/ao3_testing_sample.csv
    python3 ao3_enrich_sample.py --cookie-file ao3_cookie.txt
"""

import argparse
import csv
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Enrich AO3 sample CSV via Playwright.")
parser.add_argument("--csv", default="ao3_testing_sample.csv", help="Input/output CSV file.")
parser.add_argument("--cookie-file", default="ao3_cookie.txt", help="AO3 cookie file.")
ARGS = parser.parse_args()

# ── Constants ────────────────────────────────────────────────
TAG_SEPARATOR = " | "   # NOT a comma. This is the whole point.

ERROR_LOG = "enrich_error_log.txt"
REQUEST_GAP = (10.0, 15.0)
RETRY_AFTER_GRACE = 5
CHALLENGE_WAIT = 30
MAX_CONSECUTIVE_FAILURES = 5
ORIGIN_ERRORS = frozenset({520, 521, 522, 523, 524, 525, 526, 527})

COOKIE_DOMAIN = ".archiveofourown.org"
DEFAULT_COOKIES = [
    {"name": "view_adult", "value": "true", "domain": COOKIE_DOMAIN, "path": "/"},
]

HEADLESS_UA_TOKEN = "HeadlessChrome"
HEADLESS = os.environ.get("AO3_HEADLESS", "").strip().lower() in {"1", "true", "yes"}
BROWSER_PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ao3_browser_profile")

# CSV columns in the exact order the user specified
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


# ── Error logging ────────────────────────────────────────────
def log_error(msg, exc=None):
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[{ts}] {msg}\n")
            if exc:
                f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# CSV I/O  (strips whitespace from headers + values on read)
# ══════════════════════════════════════════════════════════════

def read_csv(path):
    """Read CSV, stripping whitespace from all headers and values."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        rows = []
        for raw in reader:
            cleaned = {}
            for k, v in raw.items():
                key = k.strip() if k else k
                val = v.strip() if isinstance(v, str) else (v or "")
                cleaned[key] = val
            rows.append(cleaned)
    return rows


def save_csv(path, rows):
    """Write CSV with clean headers. Uses csv.QUOTE_ALL so every field
    is explicitly quoted, preventing any stray commas from breaking
    the structure (belt AND suspenders with the pipe separator)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════
# BROWSER SESSION (same pattern as ao3statswithplaywright.py)
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
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        if self._cookies:
            self._ctx.add_cookies(self._cookies)
        self._warm_up()

    def _launch(self):
        from playwright.sync_api import Error as PwError
        ctx = self._launch_ctx()
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        ua = self._fix_ua(page)
        if ua is None:
            return ctx, page
        ctx.close()
        ctx = self._launch_ctx(user_agent=ua)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        return ctx, page

    def _fix_ua(self, page):
        from playwright.sync_api import Error as PwError
        try:
            ua = page.evaluate("navigator.userAgent")
        except PwError:
            return None
        if not ua or HEADLESS_UA_TOKEN not in ua:
            return None
        return ua.replace(HEADLESS_UA_TOKEN, "Chrome")

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
                os.path.join(BROWSER_PROFILE, "chrome"), channel="chrome", **opts
            )
        except PwError:
            return self._pw.chromium.launch_persistent_context(
                os.path.join(BROWSER_PROFILE, "chromium"), **opts
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
            self._page.goto("https://archiveofourown.org/", wait_until="domcontentloaded")
            self._settle()
        except (PwError, PwTimeout) as e:
            log_error("Warm up failed.", e)

    def _settle(self):
        from playwright.sync_api import Error as PwError
        try:
            body = self._page.content()
        except PwError:
            time.sleep(2)
            body = self._page.content()
        if not _is_challenge(body):
            return body
        print("    Interstitial detected. Waiting for it to clear...")
        deadline = time.monotonic() + CHALLENGE_WAIT
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                body = self._page.content()
            except PwError:
                continue
            if not _is_challenge(body):
                print("    Interstitial cleared.")
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
        try:
            resp = self._page.goto(url, wait_until="domcontentloaded",
                                   timeout=int(timeout * 1000))
        except PwTimeout:
            raise
        except PwError as e:
            msg = str(e).lower()
            if any(m in msg for m in ("closed", "crash", "disconnected")):
                log_error("Browser died; rebuilding.", e)
                self.close()
            raise
        status = resp.status if resp else 0
        reason = resp.status_text if resp else ""
        try:
            headers = resp.all_headers() if resp else {}
        except PwError:
            headers = {}
        body = self._settle()
        self._last_req = time.monotonic()
        if status == 200 and _is_challenge(body):
            status, reason = 403, "Cloudflare challenge"
        return _Resp(status, body, headers, reason, self._page.url)

    def close(self):
        for obj in (self._ctx,):
            if obj:
                try: obj.close()
                except: pass
        if self._pw:
            try: self._pw.stop()
            except: pass
        self._pw = self._ctx = self._page = None


class _Resp:
    def __init__(self, status_code, text, headers, reason, url):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.reason = reason
        self.url = url


def _is_challenge(body):
    if not body:
        return False
    low = body.lower()
    if 'id="header"' in low or 'class="work blurb' in low or "/users/logout" in low:
        return False
    if any(m in low for m in ("cf-browser-verification", "cf_chl_opt",
                               "challenge-platform", "__cf_chl")):
        return True
    title_m = re.search(r"<title[^>]*>(.*?)</title>", low, re.DOTALL)
    title = title_m.group(1).strip() if title_m else ""
    return any(m in title for m in ("just a moment", "attention required",
                                      "checking your browser"))


def _parse_retry_after(val):
    if not val:
        return None
    val = val.strip()
    try:
        return max(0, int(float(val)))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, int((dt - datetime.now(timezone.utc)).total_seconds()))
    except (TypeError, ValueError):
        return None


def _get_with_retry(browser, url, max_retries=3):
    from playwright.sync_api import Error as PwError, TimeoutError as PwTimeout
    resp = None
    for attempt in range(max_retries):
        try:
            resp = browser.get(url, timeout=60)
        except (PwError, PwTimeout) as e:
            log_error(f"Network error for {url}", e)
            wait = 10 * (attempt + 1)
            print(f"    Network error. Waiting {wait}s ({attempt+1}/{max_retries})...")
            time.sleep(wait)
            resp = None
            continue
        if resp.status_code == 429:
            ra = _parse_retry_after(
                resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            )
            wait = (ra + RETRY_AFTER_GRACE) if ra else 15 * (attempt + 1)
            print(f"    429 rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code in ORIGIN_ERRORS or resp.status_code == 0:
            wait = 20 * (attempt + 1)
            print(f"    HTTP {resp.status_code} (origin error). Waiting {wait}s...")
            time.sleep(wait)
            continue
        return resp
    return resp


def _parse_cookie_file(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    raw = re.sub(r"^\s*cookie\s*:\s*", "", raw, flags=re.IGNORECASE)
    cookies = {}
    sess = re.search(r"_otwarchive_session\s*=\s*([^;]+)", raw, re.IGNORECASE)
    if sess:
        cookies["_otwarchive_session"] = sess.group(1).strip().strip("\"'")
        raw = raw[:sess.start()] + raw[sess.end():]
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,60}", k):
            cookies[k] = v.strip().strip("\"'")
    if not cookies and raw:
        cookies["_otwarchive_session"] = raw
    return cookies


# ══════════════════════════════════════════════════════════════
# AO3 WORK PAGE PARSER
# ══════════════════════════════════════════════════════════════
#
# On archiveofourown.org/works/XXXXX, the tag metadata lives in:
#
#   <dl class="work meta group">
#     <dd class="rating tags">       → rating
#     <dd class="warning tags">      → archive warnings
#     <dd class="category tags">     → category (M/M, F/F, etc.)
#     <dd class="relationship tags"> → relationship tags
#     <dd class="character tags">    → character tags
#     <dd class="freeform tags">     → additional/freeform tags
#   </dl>
#
# Each <dd> contains <ul class="commas"> with <li><a> elements.
# BeautifulSoup class_ matching is partial, so class_="relationship"
# matches <dd class="relationship tags">.
#
# Stats live in <dl class="stats"> with <dd class="hits">, etc.
# ══════════════════════════════════════════════════════════════

def _extract_tags(soup, dd_class):
    """Extract all tag texts from a <dd class="[dd_class] tags"> element.
    Returns them joined with TAG_SEPARATOR (pipe), NOT commas."""
    node = soup.find("dd", class_=dd_class)
    if node is None:
        return ""
    links = [a.get_text(" ", strip=True) for a in node.select("li a")]
    if links:
        return TAG_SEPARATOR.join(links)
    # Fallback: plain text if no <li><a> structure
    text = node.get_text(" ", strip=True)
    return text


def _extract_stat(stats_dl, class_name):
    node = stats_dl.find("dd", class_=class_name)
    if node is None:
        return 0
    text = node.get_text(strip=True).replace(",", "")
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0


def _extract_work_id_from_url(url):
    m = re.search(r"/works/(\d+)", url)
    return m.group(1) if m else ""


def scrape_work(browser, url):
    """Visit a single AO3 work page and extract ALL metadata.
    Returns a dict with every field, or {"_error": "..."} on failure."""
    from bs4 import BeautifulSoup

    resp = _get_with_retry(browser, url)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "no response"
        return {"_error": f"HTTP {code}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Verify we got a real work page
    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
        return {"_error": "no stats found (restricted/deleted?)"}

    result = {}

    # Work ID from URL
    result["work_id"] = _extract_work_id_from_url(url)
    result["url"] = url

    # Title: <h2 class="title heading">
    title_node = soup.find("h2", class_="title")
    result["title"] = title_node.get_text(" ", strip=True) if title_node else ""

    # Fandom: first fandom tag from <dd class="fandom tags">
    fandom_links = soup.select("dd.fandom.tags ul.commas li a")
    if fandom_links:
        result["fandom"] = fandom_links[0].get_text(" ", strip=True)
    else:
        result["fandom"] = ""

    # All tag types (using PIPE separator, not commas)
    result["rating"]        = _extract_tags(soup, "rating")
    result["warnings"]      = _extract_tags(soup, "warning")
    result["category"]      = _extract_tags(soup, "category")
    result["relationships"] = _extract_tags(soup, "relationship")
    result["characters"]    = _extract_tags(soup, "character")
    result["freeform_tags"] = _extract_tags(soup, "freeform")

    # Dates from the stats block
    pub_node = stats_dl.find("dd", class_="published")
    upd_node = stats_dl.find("dd", class_="status")
    result["date_published"] = pub_node.get_text(" ", strip=True) if pub_node else ""
    result["date_updated"]   = upd_node.get_text(" ", strip=True) if upd_node else result["date_published"]

    # Numeric stats
    result["hits"]      = _extract_stat(stats_dl, "hits")
    result["kudos"]     = _extract_stat(stats_dl, "kudos")
    result["bookmarks"] = _extract_stat(stats_dl, "bookmarks")
    result["comments"]  = _extract_stat(stats_dl, "comments")
    result["words"]     = _extract_stat(stats_dl, "words")

    ch_node = stats_dl.find("dd", class_="chapters")
    if ch_node:
        ch_m = re.search(r"(\d+)\s*/", ch_node.get_text(" ", strip=True))
        result["chapters"] = int(ch_m.group(1)) if ch_m else 1
    else:
        result["chapters"] = 1

    lang_node = soup.find("dd", class_="language")
    result["language"] = lang_node.get_text(" ", strip=True) if lang_node else ""

    # Engagement ratios
    h, k, b, c, w = result["hits"], result["kudos"], result["bookmarks"], result["comments"], result["words"]
    result["kudos_to_hits"]          = round(k / h * 100, 4) if h > 0 else 0
    result["bookmarks_to_hits"]      = round(b / h * 100, 4) if h > 0 else 0
    result["comments_to_hits"]       = round(c / h * 100, 4) if h > 0 else 0
    result["comments_to_kudos"]      = round(c / k * 100, 4) if k > 0 else 0
    result["kudos_per_10k_words"]    = round(k / w * 10000, 4) if w > 0 else 0
    result["bookmarks_per_10k_words"]= round(b / w * 10000, 4) if w > 0 else 0
    result["comments_per_10k_words"] = round(c / w * 10000, 4) if w > 0 else 0

    return result


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    csv_path = ARGS.csv

    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found.")

    # Read existing CSV (with whitespace stripping)
    rows = read_csv(csv_path)
    print(f"Read {len(rows)} rows from {csv_path}")

    if rows:
        print(f"Columns found: {list(rows[0].keys())}")

    # Figure out which rows need enrichment
    todo = []
    for i, r in enumerate(rows):
        enriched = r.get("tags_enriched", "").strip().lower()
        if enriched == "yes":
            continue
        url = r.get("url", "").strip()
        if url and "archiveofourown.org/works/" in url:
            todo.append(i)

    already_done = len(rows) - len(todo)
    print(f"\n  Already enriched: {already_done}")
    print(f"  Still need enrichment: {len(todo)}")

    if not todo:
        print("\nAll works already enriched! Nothing to do.")
        return

    # Start the browser
    print(f"\nStarting browser to enrich {len(todo)} works...")
    print(f"Rate: ~1 work every 10 to 15 seconds (respecting AO3 limits).\n")

    browser = BrowserSession()

    # Load cookies if available
    cookies = _parse_cookie_file(ARGS.cookie_file)
    if cookies:
        browser.add_cookies(cookies)
        print(f"Loaded cookies: {', '.join(sorted(cookies.keys()))}")

    browser.start()
    consecutive_failures = 0

    try:
        for progress, idx in enumerate(todo, start=1):
            row = rows[idx]
            url = row.get("url", "").strip()
            sid = row.get("sample_id", "?")

            print(f"  [{progress}/{len(todo)}] #{sid} {url}: ", end="", flush=True)

            try:
                data = scrape_work(browser, url)
            except Exception as e:
                log_error(f"Exception scraping {url}", e)
                print(f"exception ({e.__class__.__name__}). Skipping.")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\nToo many consecutive failures. Stopping.")
                    print("Re-run to resume from where it stopped.")
                    break
                continue

            if "_error" in data:
                print(f"{data['_error']}. Skipping.")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\nToo many consecutive failures. Stopping.")
                    print("Re-run to resume from where it stopped.")
                    break
                continue

            consecutive_failures = 0

            # Overwrite ALL fields from the scraped data (except sample_id)
            for key in CSV_FIELDS:
                if key == "sample_id":
                    continue  # keep the original sample_id
                if key == "tags_enriched":
                    continue  # set below
                if key in data:
                    row[key] = data[key]

            row["tags_enriched"] = "yes"
            rows[idx] = row

            # Quick summary
            n_rels  = len(data.get("relationships", "").split(TAG_SEPARATOR)) if data.get("relationships") else 0
            n_chars = len(data.get("characters", "").split(TAG_SEPARATOR)) if data.get("characters") else 0
            n_free  = len(data.get("freeform_tags", "").split(TAG_SEPARATOR)) if data.get("freeform_tags") else 0
            print(f"done  rels={n_rels}  chars={n_chars}  freeform={n_free}")

            # SAVE AFTER EVERY WORK (progress never lost)
            save_csv(csv_path, rows)

    except KeyboardInterrupt:
        print("\n\nStopped by user. Progress saved.")
        save_csv(csv_path, rows)

    finally:
        browser.close()

    done = sum(1 for r in rows if r.get("tags_enriched", "").strip().lower() == "yes")
    print(f"\nComplete: {done}/{len(rows)} works enriched.")
    print(f"Saved to {csv_path}")
    print(f"\nTag separator used: \"{TAG_SEPARATOR}\"")
    print("To split tags in Python:  row['relationships'].split(' | ')")
    print("To split tags in Excel:   use Text to Columns with delimiter |")


if __name__ == "__main__":
    main()

