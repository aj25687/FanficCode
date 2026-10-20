
#!/usr/bin/env python3
"""
ao3_stratified_sampler.py
=========================
Two-phase stratified sampler for hand-coding AO3 works.

Phase 1  (offline, instant):
  Reads ao3_results.csv, computes a proportionally stratified sample
  (95% CI, 5% ME, M/M / F/F / F/M shares match population), and
  saves hand_coding_sample.csv immediately.

Phase 2  (online, Playwright):
  Visits each sampled work's individual page on AO3 to fill in the
  FULL tag set (relationships, characters, freeform/additional tags,
  date_published) that search-result blurbs don't expose.
  Saves after every work so progress is never lost.

Usage:
    python3 ao3_stratified_sampler.py                           # defaults
    python3 ao3_stratified_sampler.py path/to/ao3_results.csv   # custom CSV
    python3 ao3_stratified_sampler.py --seed 42                 # reproducible
    python3 ao3_stratified_sampler.py --skip-scrape             # Phase 1 only
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

# Playwright is only imported when Phase 2 runs
# (so --skip-scrape works without it installed)

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Stratified sample from ao3_results.csv, then enrich via Playwright."
)
parser.add_argument(
    "csv", nargs="?", default="ao3_results.csv",
    help="Path to ao3_results.csv (default: current directory)."
)
parser.add_argument(
    "--confidence", type=float, default=0.95,
    help="Confidence level (default 0.95)."
)
parser.add_argument(
    "--margin", type=float, default=0.05,
    help="Margin of error (default 0.05)."
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Random seed for reproducibility."
)
parser.add_argument(
    "--out", default="hand_coding_sample.csv",
    help="Output CSV filename (default: hand_coding_sample.csv)."
)
parser.add_argument(
    "--skip-scrape", action="store_true",
    help="Save the sample CSV (Phase 1) but skip the Playwright tag enrichment (Phase 2)."
)
parser.add_argument(
    "--enrich-only", action="store_true",
    help="Skip sampling; only run Phase 2 on an existing hand_coding_sample.csv."
)
parser.add_argument(
    "--cookie-file", default="ao3_cookie.txt",
    help="Path to ao3_cookie.txt for logged-in access (default: ao3_cookie.txt)."
)
ARGS = parser.parse_args()


# ── Constants ────────────────────────────────────────────────
Z_SCORES = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
Z = Z_SCORES.get(ARGS.confidence, 1.96)
E = ARGS.margin

ERROR_LOG = "sampler_error_log.txt"

CATEGORY_TOKENS = {
    "f/f": "F/F", "m/m": "M/M", "f/m": "F/M",
    "gen": "Gen", "multi": "Multi", "other": "Other",
}
NO_CATEGORY_TOKENS = {"", "no category", "none"}
TARGET_STRATA = ["F/F", "M/M", "F/M"]

# AO3 individual work page: the <dd> CSS classes for each tag type.
# When you visit archiveofourown.org/works/XXXXX, the tags live in a
# <dl class="work meta group"> with these <dd> elements:
#   <dd class="rating tags">       → rating
#   <dd class="warning tags">      → archive warnings
#   <dd class="category tags">     → category (M/M, F/F, etc.)
#   <dd class="relationship tags"> → relationship tags
#   <dd class="character tags">    → character tags
#   <dd class="freeform tags">     → additional / freeform tags
# BeautifulSoup's class_ matching is partial, so class_="relationship"
# matches <dd class="relationship tags">.
TAG_DD_CLASSES = {
    "rating":       "rating",
    "warnings":     "warning",
    "category":     "category",
    "relationships":"relationship",
    "characters":   "character",
    "freeform_tags":"freeform",
}

# Pacing: same as the main scraper (10-15 s between page loads)
REQUEST_GAP_SECONDS = (10.0, 15.0)
RETRY_AFTER_GRACE = 5
CHALLENGE_WAIT = 30
MAX_CONSECUTIVE_FAILURES = 3
ORIGIN_ERROR_STATUSES = frozenset({520, 521, 522, 523, 524, 525, 526, 527})

COOKIE_DOMAIN = ".archiveofourown.org"
DEFAULT_COOKIES = [
    {"name": "view_adult", "value": "true", "domain": COOKIE_DOMAIN, "path": "/"},
]

HEADLESS_UA_TOKEN = "HeadlessChrome"
HEADLESS = os.environ.get("AO3_HEADLESS", "").strip().lower() in {"1", "true", "yes"}
BROWSER_PROFILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ao3_browser_profile"
)

# ── Output CSV columns (in order) ───────────────────────────
OUTPUT_FIELDS = [
    "sample_id",
    "work_id",
    "url",
    "title",
    "fandom",
    "fandoms",
    "stratum",
    # Tag columns (populated fully in Phase 2)
    "rating",
    "warnings",
    "category",
    "relationships",
    "characters",
    "freeform_tags",
    # Dates
    "date_published",
    "date_updated",
    # Stats
    "words",
    "chapters",
    "language",
    "hits",
    "kudos",
    "bookmarks",
    "comments",
    # Engagement ratios
    "kudos_to_hits",
    "bookmarks_to_hits",
    "comments_to_hits",
    "comments_to_kudos",
    "kudos_per_10k_words",
    "bookmarks_per_10k_words",
    "comments_per_10k_words",
    # Phase 2 status
    "tags_enriched",
    # Blank hand-coding columns
    "your_code_1",
    "your_code_2",
    "notes",
]


# ── Error logging ────────────────────────────────────────────
def log_error(message, exception=None):
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"\n[{ts}] {message}\n")
            if exception:
                f.write("".join(traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# PHASE 1: STRATIFIED SAMPLING
# ══════════════════════════════════════════════════════════════

def parse_categories(category_text, warnings_text):
    """Read category labels from both the category and warnings columns.
    (The scraper sometimes places category info in warnings and vice versa,
    as noted in your summary.txt: '11090 works had category labels in the
    warnings column'.)"""
    cats = []
    for text in (category_text, warnings_text):
        if not isinstance(text, str):
            continue
        for token in text.split(","):
            key = token.strip().lower()
            if key in NO_CATEGORY_TOKENS:
                continue
            if key in CATEGORY_TOKENS:
                name = CATEGORY_TOKENS[key]
                if name not in cats:
                    cats.append(name)
    return cats


def cochran_sample_size(N, z=1.96, e=0.05, p=0.5):
    """Cochran formula with finite population correction."""
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / N)
    return math.ceil(n)


def run_phase1(csv_path, out_path, seed):
    """Sample from ao3_results.csv and save hand_coding_sample.csv."""
    from pathlib import Path

    if not Path(csv_path).exists():
        sys.exit(f"ERROR: cannot find {csv_path}.")

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype={"work_id": str})

    if "role" in df.columns:
        df = df[df["role"] == "cohort"].copy()

    n_raw = len(df)
    df = df.dropna(subset=["work_id"]).drop_duplicates(subset=["work_id"], keep="first")
    print(f"  Raw cohort rows: {n_raw}; unique works after dedup: {len(df)}")

    # Parse categories from both columns
    df["_cats"] = [
        parse_categories(c, w)
        for c, w in zip(df["category"], df.get("warnings", pd.Series(dtype=str)))
    ]

    for cat in TARGET_STRATA:
        df[f"has_{cat}"] = df["_cats"].apply(lambda cats, c=cat: c in cats)

    # Assign each work to ONE stratum (rarest first so smallest group is protected)
    stratum_sizes = {cat: df[f"has_{cat}"].sum() for cat in TARGET_STRATA}
    priority = sorted(TARGET_STRATA, key=lambda c: stratum_sizes[c])

    def assign_stratum(cats):
        for cat in priority:
            if cat in cats:
                return cat
        return "Other"

    df["stratum"] = df["_cats"].apply(assign_stratum)

    pop_counts = df["stratum"].value_counts()
    N_total = len(df)

    print(f"\n{'='*55}")
    print("POPULATION BREAKDOWN")
    print(f"{'='*55}")
    for s in TARGET_STRATA + ["Other"]:
        n = pop_counts.get(s, 0)
        pct = n / N_total * 100
        print(f"  {s:>6}: {n:>6,} works  ({pct:5.1f}%)")
    print(f"  {'TOTAL':>6}: {N_total:>6,} works")

    # Cochran sample size
    n_needed = cochran_sample_size(N_total, z=Z, e=E)
    print(f"\nCochran sample size (95% CI, 5% ME, N={N_total:,}): {n_needed}")

    # Proportional allocation
    strata_list = TARGET_STRATA + (["Other"] if pop_counts.get("Other", 0) > 0 else [])
    stratum_alloc = {}
    allocated = 0
    for i, s in enumerate(strata_list):
        pop_n = pop_counts.get(s, 0)
        if pop_n == 0:
            stratum_alloc[s] = 0
            continue
        if i == len(strata_list) - 1:
            stratum_alloc[s] = n_needed - allocated
        else:
            alloc = round(n_needed * (pop_n / N_total))
            stratum_alloc[s] = min(alloc, pop_n)
            allocated += stratum_alloc[s]

    # Guarantee at least 2 per target stratum
    for s in TARGET_STRATA:
        pop_n = pop_counts.get(s, 0)
        if pop_n > 0 and stratum_alloc.get(s, 0) < 2:
            stratum_alloc[s] = min(2, pop_n)

    print(f"\n{'='*55}")
    print("SAMPLE ALLOCATION (proportional)")
    print(f"{'='*55}")
    total_sample = 0
    for s in strata_list:
        alloc = stratum_alloc.get(s, 0)
        pop_n = pop_counts.get(s, 0)
        pct = alloc / n_needed * 100 if n_needed else 0
        pop_pct = pop_n / N_total * 100
        print(f"  {s:>6}: draw {alloc:>4} of {pop_n:>6,}  "
              f"(sample {pct:5.1f}% | population {pop_pct:5.1f}%)")
        total_sample += alloc
    print(f"  {'TOTAL':>6}: {total_sample:>4} works to hand code")

    # Draw the sample
    rng = np.random.default_rng(seed)
    pieces = []
    for s in strata_list:
        pool = df[df["stratum"] == s]
        n_draw = stratum_alloc.get(s, 0)
        if n_draw == 0 or pool.empty:
            continue
        chosen = pool.sample(n=n_draw, random_state=rng.integers(0, 2**31))
        pieces.append(chosen)

    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.sample(frac=1, random_state=rng.integers(0, 2**31)).reset_index(drop=True)

    # Build output
    rows = []
    for idx, row in sample.iterrows():
        wid = str(row["work_id"])
        out_row = {
            "sample_id": idx + 1,
            "work_id": wid,
            "url": f"https://archiveofourown.org/works/{wid}",
            "title": row.get("title", ""),
            "fandom": row.get("fandom", ""),
            "fandoms": row.get("fandoms", ""),
            "stratum": row["stratum"],
            "rating": row.get("rating", ""),
            "warnings": row.get("warnings", ""),
            "category": row.get("category", ""),
            "relationships": row.get("relationships", ""),
            "characters": "",          # filled in Phase 2
            "freeform_tags": "",       # filled in Phase 2
            "date_published": row.get("date_published", ""),
            "date_updated": row.get("date_updated", ""),
            "words": row.get("words", ""),
            "chapters": row.get("chapters", ""),
            "language": row.get("language", ""),
            "hits": row.get("hits", ""),
            "kudos": row.get("kudos", ""),
            "bookmarks": row.get("bookmarks", ""),
            "comments": row.get("comments", ""),
            "kudos_to_hits": row.get("kudos_to_hits", ""),
            "bookmarks_to_hits": row.get("bookmarks_to_hits", ""),
            "comments_to_hits": row.get("comments_to_hits", ""),
            "comments_to_kudos": row.get("comments_to_kudos", ""),
            "kudos_per_10k_words": row.get("kudos_per_10k_words", ""),
            "bookmarks_per_10k_words": row.get("bookmarks_per_10k_words", ""),
            "comments_per_10k_words": row.get("comments_per_10k_words", ""),
            "tags_enriched": "no",
            "your_code_1": "",
            "your_code_2": "",
            "notes": "",
        }
        rows.append(out_row)

    # Write the CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Phase 1 complete: saved {len(rows)} works to {out_path}")

    # Verification
    out_df = pd.DataFrame(rows)
    print(f"\n{'='*55}")
    print("VERIFICATION: does the sample match the population?")
    print(f"{'='*55}")
    print(f"  {'Stratum':>8}  {'Pop %':>7}  {'Sample %':>9}  {'Diff':>6}")
    for s in strata_list:
        pop_pct = pop_counts.get(s, 0) / N_total * 100
        samp_pct = (out_df["stratum"] == s).sum() / len(out_df) * 100
        diff = samp_pct - pop_pct
        flag = " ⚠" if abs(diff) > 2 else ""
        print(f"  {s:>8}  {pop_pct:6.1f}%  {samp_pct:8.1f}%  {diff:+5.1f}%{flag}")
    print(f"\nTotal: {len(rows)} works | CI: {ARGS.confidence*100:.0f}% | ME: ±{ARGS.margin*100:.0f}%")
    if seed is not None:
        print(f"Seed: {seed}")

    return len(rows)


# ══════════════════════════════════════════════════════════════
# PHASE 2: PLAYWRIGHT TAG ENRICHMENT
# ══════════════════════════════════════════════════════════════

class BrowserSession:
    """Minimal Playwright browser session, matching the rate limiting
    and interstitial handling of ao3statswithplaywright.py."""

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
        gap = random.uniform(*REQUEST_GAP_SECONDS)
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
            log_error("Warm-up failed.", e)

    def _settle(self):
        """Wait for Cloudflare interstitial to clear."""
        from playwright.sync_api import Error as PwError
        try:
            body = self._page.content()
        except PwError:
            time.sleep(2)
            body = self._page.content()
        if not _looks_like_challenge(body):
            return body
        print("  Interstitial detected. Waiting for it to clear"
              f"{'' if HEADLESS else ' (solve it in the browser window if needed)'}...")
        deadline = time.monotonic() + CHALLENGE_WAIT
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                body = self._page.content()
            except PwError:
                continue
            if not _looks_like_challenge(body):
                print("  Interstitial cleared.")
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
        if status == 200 and _looks_like_challenge(body):
            status, reason = 403, "Cloudflare challenge"
        return _Response(status, body, headers, reason, self._page.url)

    def close(self):
        for obj in (self._ctx, ):
            if obj:
                try: obj.close()
                except: pass
        if self._pw:
            try: self._pw.stop()
            except: pass
        self._pw = self._ctx = self._page = None


class _Response:
    def __init__(self, status_code, text, headers, reason, url):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.reason = reason
        self.url = url


def _looks_like_challenge(body):
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
            print(f"  Network error. Waiting {wait}s ({attempt+1}/{max_retries})...")
            time.sleep(wait)
            resp = None
            continue
        if resp.status_code == 429:
            ra = _parse_retry_after(
                resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            )
            wait = (ra + RETRY_AFTER_GRACE) if ra else 15 * (attempt + 1)
            print(f"  429 rate-limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code in ORIGIN_ERROR_STATUSES or resp.status_code == 0:
            wait = 20 * (attempt + 1)
            print(f"  HTTP {resp.status_code} (origin error). Waiting {wait}s...")
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


def _extract_meta_tags(soup, dd_class):
    """Extract tag text from a <dd class="[dd_class] tags"> element
    on an AO3 individual work page.  Each tag is an <a> inside an
    <li> inside a <ul class="commas">."""
    from bs4 import BeautifulSoup as _BS
    node = soup.find("dd", class_=dd_class)
    if node is None:
        return ""
    links = [a.get_text(" ", strip=True) for a in node.select("li a")]
    if links:
        return ", ".join(links)
    return node.get_text(" ", strip=True)


def _extract_stat(stats_dl, class_name):
    node = stats_dl.find("dd", class_=class_name)
    if node is None:
        return 0
    text = node.get_text(strip=True).replace(",", "")
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0


def _scrape_work_tags(browser, work_id):
    """Visit archiveofourown.org/works/{work_id} and return a dict
    of the full tag set plus date_published."""
    from bs4 import BeautifulSoup

    url = f"https://archiveofourown.org/works/{work_id}"
    resp = _get_with_retry(browser, url)

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "no response"
        return {"_error": f"HTTP {code}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Verify we actually got a work page
    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
        return {"_error": "no stats found (may be restricted/deleted)"}

    result = {}

    # Full tags from the work meta section
    for col_name, dd_class in TAG_DD_CLASSES.items():
        result[col_name] = _extract_meta_tags(soup, dd_class)

    # Title
    title_node = soup.find("h2", class_="title")
    if title_node:
        result["title"] = title_node.get_text(" ", strip=True)

    # Fandoms
    fandom_links = soup.select("dd.fandom.tags ul.commas li a")
    if fandom_links:
        result["fandoms"] = ", ".join(
            a.get_text(" ", strip=True) for a in fandom_links
        )
        result["fandom"] = fandom_links[0].get_text(" ", strip=True)

    # Date published / updated (from the stats block)
    pub_node = stats_dl.find("dd", class_="published")
    upd_node = stats_dl.find("dd", class_="status")
    if pub_node:
        result["date_published"] = pub_node.get_text(" ", strip=True)
    if upd_node:
        result["date_updated"] = upd_node.get_text(" ", strip=True)

    # Refresh stats while we're here
    result["hits"] = _extract_stat(stats_dl, "hits")
    result["kudos"] = _extract_stat(stats_dl, "kudos")
    result["bookmarks"] = _extract_stat(stats_dl, "bookmarks")
    result["comments"] = _extract_stat(stats_dl, "comments")
    result["words"] = _extract_stat(stats_dl, "words")

    ch_node = stats_dl.find("dd", class_="chapters")
    if ch_node:
        ch_m = re.search(r"(\d+)\s*/", ch_node.get_text(" ", strip=True))
        result["chapters"] = int(ch_m.group(1)) if ch_m else 1

    lang_node = soup.find("dd", class_="language")
    if lang_node:
        result["language"] = lang_node.get_text(" ", strip=True)

    return result


def _recompute_ratios(row):
    """Recalculate engagement ratios from the refreshed stats."""
    h = int(row.get("hits") or 0)
    k = int(row.get("kudos") or 0)
    b = int(row.get("bookmarks") or 0)
    c = int(row.get("comments") or 0)
    w = int(row.get("words") or 0)
    row["kudos_to_hits"] = round(k / h * 100, 4) if h > 0 else 0
    row["bookmarks_to_hits"] = round(b / h * 100, 4) if h > 0 else 0
    row["comments_to_hits"] = round(c / h * 100, 4) if h > 0 else 0
    row["comments_to_kudos"] = round(c / k * 100, 4) if k > 0 else 0
    row["kudos_per_10k_words"] = round(k / w * 10000, 4) if w > 0 else 0
    row["bookmarks_per_10k_words"] = round(b / w * 10000, 4) if w > 0 else 0
    row["comments_per_10k_words"] = round(c / w * 10000, 4) if w > 0 else 0
    return row


def _save_rows(out_path, rows):
    """Overwrite the CSV with current rows (preserves hand-coding work)."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_phase2(out_path, cookie_file):
    """Visit each work's page to fill in full tags."""
    if not os.path.exists(out_path):
        sys.exit(f"ERROR: {out_path} not found. Run Phase 1 first.")

    # Read existing sample
    with open(out_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    todo = [i for i, r in enumerate(rows) if r.get("tags_enriched", "no") != "yes"]
    if not todo:
        print("All works already enriched. Nothing to do.")
        return

    print(f"\nPhase 2: {len(todo)} works need tag enrichment.")
    print("Starting Playwright browser...\n")

    browser = BrowserSession()

    # Load cookies
    cookies = _parse_cookie_file(cookie_file)
    if cookies:
        browser.add_cookies(cookies)
        print(f"  Loaded cookies from {cookie_file}: {', '.join(sorted(cookies.keys()))}")

    browser.start()
    consecutive_failures = 0

    try:
        for progress, idx in enumerate(todo, start=1):
            row = rows[idx]
            wid = row["work_id"]
            print(f"  [{progress}/{len(todo)}] Work {wid}: ", end="", flush=True)

            try:
                enriched = _scrape_work_tags(browser, wid)
            except Exception as e:
                log_error(f"Exception scraping work {wid}", e)
                print(f"exception ({e.__class__.__name__}). Skipping.")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\nToo many consecutive failures. Stopping Phase 2.")
                    print("Re-run later; already-enriched works are saved.")
                    break
                continue

            if "_error" in enriched:
                print(f"{enriched['_error']}. Skipping.")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("\nToo many consecutive failures. Stopping Phase 2.")
                    break
                continue

            consecutive_failures = 0

            # Merge enriched data into the row
            for key, val in enriched.items():
                if key.startswith("_"):
                    continue
                if key in row:
                    row[key] = val

            row = _recompute_ratios(row)
            row["tags_enriched"] = "yes"
            rows[idx] = row

            rels = enriched.get("relationships", "")
            chars = enriched.get("characters", "")
            print(f"✓  rels={len(rels.split(',')) if rels else 0}  "
                  f"chars={len(chars.split(',')) if chars else 0}  "
                  f"freeform={len(enriched.get('freeform_tags','').split(',')) if enriched.get('freeform_tags') else 0}")

            # Save after every work so progress is never lost
            _save_rows(out_path, rows)

    finally:
        browser.close()

    done = sum(1 for r in rows if r.get("tags_enriched") == "yes")
    print(f"\n✓ Phase 2 complete: {done}/{len(rows)} works enriched.")
    print(f"  Saved to {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    if not ARGS.enrich_only:
        run_phase1(ARGS.csv, ARGS.out, ARGS.seed)
    else:
        print("--enrich-only: skipping Phase 1 (sampling).")

    if not ARGS.skip_scrape:
        print("\n" + "="*55)
        print("PHASE 2: TAG ENRICHMENT VIA PLAYWRIGHT")
        print("="*55)
        print("This will open a browser and visit each work's page")
        print("to get the full tags (relationships, characters, freeform).")
        print(f"Rate: ~1 work every 10-15 seconds (respecting AO3's limits).\n")

        run_phase2(ARGS.out, ARGS.cookie_file)
    else:
        print("\n--skip-scrape: Phase 2 skipped.")
        print(f"Run later with:  python3 {sys.argv[0]} --enrich-only")

    print("\nOutput columns:")
    for c in OUTPUT_FIELDS:
        print(f"  • {c}")
    print("\nHappy coding! 🎉")


if __name__ == "__main__":
    main()

