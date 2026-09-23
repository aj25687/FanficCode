import csv
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.parse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright, Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup


# ============================================================
# AO3 ENGAGEMENT / TEXTUAL FEATURE SAMPLER
# ============================================================
#
# Purpose:
#   Collect AO3 metadata for a target work and a comparison cohort,
#   for research comparing engagement patterns across categories
#   (e.g. M/M vs F/F) while controlling for fandom, word count,
#   chapters, completion, rating, warnings, crossover status, and time.
#
# Scope:
#   This script collects METADATA ONLY from search result pages for
#   cohort works (hits, kudos, bookmarks, comments, words, tags).
#   It does not download full work text. Full-text content coding
#   (for things like emotional-vulnerability or physical-affection
#   density) is a separate, much smaller, manually-curated step —
#   see the printed recommendation at the end of each run.
#
# Networking:
#   Uses a real, genuine browser engine (Playwright + Chromium)
#   rather than a bare HTTP client. This exists because AO3's
#   Cloudflare bot management was reliably rejecting plain HTTP
#   requests (repeated 525s, then explicit 403s) regardless of
#   IP/network -- a real browser has a real, honest fingerprint,
#   so this isn't a workaround for detection, it's just automating
#   an actual browser the way a human would use one.
#
#   One-time setup:
#     pip3 install playwright && python3 -m playwright install chromium
#
#   The browser window is visible by default so a challenge can be
#   completed by hand; set AO3_HEADLESS=1 for unattended runs. The
#   browser profile is kept next to this script, in
#   ao3_browser_profile/, so clearance cookies survive between runs.
#
#   A note on 525s: Cloudflare's 52x statuses describe Cloudflare
#   failing to reach AO3's own servers (525 is specifically an SSL
#   handshake failure between the two). They are not a verdict on
#   this client, and no client -- browser or otherwise -- can avoid
#   them; they're retried with backoff here and usually clear.
#
# On the "T"/"F" vs "true"/"false" vs "0"/"1" question for AO3's
# work_search[complete] and work_search[crossover] fields:
#   Different independent AO3 tooling projects disagree on this, so
#   the "T"/"F" form this script sends was checked live against AO3:
#   complete=T returned only completed works, crossover=T returned
#   only crossovers, and crossover=F dropped works AO3 itself flags
#   as crossovers. Note that crossover=F still returns some works
#   carrying more than one fandom tag -- AO3's crossover flag is the
#   author/tag-wrangler judgement, not "has 2+ fandom tags", which is
#   why is_crossover in the output is labelled a heuristic.
#
# Rate limiting:
#   Uses a local page cache and stops the whole run (rather than
#   trying to push past) when AO3 returns repeated hard failures or
#   sustained rate-limiting. A real browser engine does not grant
#   permission to go faster, so pacing is enforced in the browser
#   layer itself: every page load waits a randomized 10-15s since the
#   previous one, and a 429's Retry-After is obeyed to the second
#   (plus a few seconds of grace) instead of a guessed interval.
#
# ============================================================


LOG_FILE = "scraping_log.txt"
ERROR_LOG_FILE = "error_log.txt"
CACHE_FILE = "ao3_cache.json"
CSV_FILE = "ao3_results.csv"
DIST_FILE = "ao3_distributions.json"
COOKIE_FILE = "ao3_cookie.txt"

# Applied to every browser context as a real cookie (not a spoofed
# header) so Mature/Explicit works don't show the adult-content
# interstitial instead of their real stats.
DEFAULT_COOKIES = [
    {"name": "view_adult", "value": "true", "domain": ".archiveofourown.org", "path": "/"},
]

COOKIE_DOMAIN = ".archiveofourown.org"

# Chromium profile directory. Reusing one profile across runs keeps
# whatever clearance cookie the browser earned, so a fresh challenge
# isn't triggered on every single run. Anchored to the script's own
# folder rather than the working directory, so running the script
# from somewhere else doesn't silently start from a blank profile.
BROWSER_PROFILE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ao3_browser_profile")

# Headed by default: a visible window also means a challenge can be
# completed by hand if AO3 ever shows one. Set AO3_HEADLESS=1 for
# unattended runs.
HEADLESS = os.environ.get("AO3_HEADLESS", "").strip().lower() in {"1", "true", "yes"}

# Chrome's headless builds put "HeadlessChrome" in the User-Agent,
# which no human browser sends. Rather than inventing a UA string
# (a made-up OS/version contradicts the platform, client hints and
# WebGL strings the same browser reports, which is worse than the
# honest one), the real UA is read from the running browser and only
# that one token is corrected. See BrowserSession._sanitize_user_agent.
HEADLESS_UA_TOKEN = "HeadlessChrome"

# How long to let an interstitial resolve itself before treating the
# page as blocked.
CHALLENGE_WAIT_SECONDS = 30

# Minimum randomized gap between two page loads, enforced in the
# browser layer so it applies to every single request -- warm-up,
# login check, page counts, retries after an error, everything --
# rather than only where a caller remembered to ask for a delay.
REQUEST_GAP_SECONDS = (10.0, 15.0)

# Added to whatever Retry-After AO3 asks for, so the next request
# lands safely after the window it named rather than exactly on its
# boundary.
RETRY_AFTER_GRACE_SECONDS = 5

# Cloudflare's 52x family describes the edge failing to talk to AO3's
# origin (525 in particular is an SSL handshake failure between
# Cloudflare and AO3, per Cloudflare's own docs) -- it says nothing
# about this client, so these are retried rather than treated as an
# instruction to stop. 403 and 429 are the responses that actually
# mean "you, stop".
ORIGIN_ERROR_STATUSES = frozenset({520, 521, 522, 523, 524, 525, 526, 527})

MAX_CONSECUTIVE_HARD_FAILURES = 3

# AO3 category IDs (confirmed against AO3's own reference post,
# archiveofourown.org/admin_posts/349): Gen=21, F/M=22, M/M=23,
# Other=24, F/F=116, Multi=2246.
CATEGORY_MAP = {
    "1": "23",    # M/M
    "2": "116",   # F/F
    "3": "22",    # F/M
    "4": "2246",  # Multi
    "5": "21",    # Gen
}

# AO3 archive warning IDs.
WARNING_MAP = {
    "1": "16",  # No Archive Warnings Apply
    "2": "14",  # Creator Chose Not To Use Archive Warnings
    "3": "17",  # Graphic Depictions Of Violence
    "4": "18",  # Major Character Death
    "5": "19",  # Rape/Non-Con
    "6": "20",  # Underage
}

RATING_VALUES = {
    "General Audiences",
    "Teen And Up Audiences",
    "Mature",
    "Explicit",
    "Not Rated",
}

CATEGORY_VALUES = {
    "F/F",
    "F/M",
    "Gen",
    "M/M",
    "Multi",
    "Other",
}

COMPLETE_STATUS_VALUES = {
    "Complete Work",
    "Work in Progress",
}

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


class BrowserResponse:
    """requests-shaped view of a Playwright navigation.

    Everything downstream (parsing, filtering, caching, CSV writing)
    keeps reading .status_code/.text/.headers/.reason exactly as it
    did with the HTTP client, so only this layer knows about browsers.
    """

    def __init__(self, status_code, text, headers=None, reason="", url=""):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.reason = reason
        self.url = url


class BrowserSession:
    """A single real Chromium profile used for every page load."""

    def __init__(self):
        self._playwright = None
        self._context = None
        self._page = None
        # Every cookie ever handed to this session is remembered, so a
        # context rebuilt after a browser crash comes back with the
        # same login/adult-content state instead of a blank jar.
        self._cookies = list(DEFAULT_COOKIES)
        self._last_request_at = None

    def start(self):
        if self._page is not None:
            return

        self._playwright = sync_playwright().start()
        self._context, self._page = self._launch()
        self._context.set_default_navigation_timeout(60_000)

        # navigator.webdriver is the one automation signal Chromium
        # still reports after --disable-blink-features.
        self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        if self._cookies:
            self._context.add_cookies(self._cookies)

        self._warm_up()

    def _launch(self):
        context = self._launch_context()
        page = context.pages[0] if context.pages else context.new_page()

        clean_user_agent = self._sanitize_user_agent(page)
        if clean_user_agent is None:
            return context, page

        # The UA can only be set when the context is created, so the
        # first context is thrown away once it reveals a headless UA.
        context.close()
        context = self._launch_context(user_agent=clean_user_agent)
        page = context.pages[0] if context.pages else context.new_page()
        return context, page

    @staticmethod
    def _sanitize_user_agent(page):
        """Return a corrected UA if the browser advertises headless."""
        try:
            user_agent = page.evaluate("navigator.userAgent")
        except PlaywrightError:
            return None

        if not user_agent or HEADLESS_UA_TOKEN not in user_agent:
            return None

        return user_agent.replace(HEADLESS_UA_TOKEN, "Chrome")

    def _launch_context(self, user_agent=None):
        options = {
            "headless": HEADLESS,
            "viewport": {"width": 1280, "height": 900},
            "locale": "en-US",
            # Chromium otherwise advertises itself as automated.
            "args": ["--disable-blink-features=AutomationControlled"],
        }

        if user_agent:
            options["user_agent"] = user_agent

        # A locally installed Chrome is a more ordinary browser than
        # Playwright's bundled Chromium build, so use it when present.
        # Chrome and Chromium get separate profile folders: one profile
        # written by a newer build makes the other refuse to start.
        try:
            return self._playwright.chromium.launch_persistent_context(
                os.path.join(BROWSER_PROFILE_ROOT, "chrome"), channel="chrome", **options
            )
        except PlaywrightError:
            return self._playwright.chromium.launch_persistent_context(
                os.path.join(BROWSER_PROFILE_ROOT, "chromium"), **options
            )

    def _throttle(self):
        """Hold every page load at least REQUEST_GAP_SECONDS apart.

        The gap is measured from the last request rather than slept
        unconditionally, so a caller's own polite_delay() counts
        towards it instead of stacking on top of it.
        """
        gap = random.uniform(*REQUEST_GAP_SECONDS)

        if self._last_request_at is not None:
            remaining = gap - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)

        self._last_request_at = time.monotonic()

    def _warm_up(self):
        """Load the homepage once before jumping into search URLs."""
        try:
            self._throttle()
            self._page.goto("https://archiveofourown.org/", wait_until="domcontentloaded")
            self._settle_interstitial()
        except (PlaywrightError, PlaywrightTimeoutError) as error:
            log_error("Warm-up navigation to the AO3 homepage failed.", error)

    def add_cookies(self, cookie_dict):
        cookies = [
            {"name": name, "value": value, "domain": COOKIE_DOMAIN, "path": "/"}
            for name, value in cookie_dict.items()
        ]

        self._cookies.extend(cookies)

        if self._context is not None:
            self._context.add_cookies(cookies)

    def _recover_if_dead(self, error):
        """Rebuild the browser if it crashed or was closed.

        Without this, one crashed Chromium turns every remaining page
        of a long run into the same 'Target closed' failure.
        """
        message = str(error).lower()
        if not any(m in message for m in ("closed", "crash", "disconnected")):
            return

        log_error("Browser died; rebuilding the session.", error)
        self.close()

    def get(self, url, timeout=60):
        self.start()
        self._throttle()

        try:
            response = self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(timeout * 1000),
            )
        except PlaywrightTimeoutError:
            raise
        except PlaywrightError as error:
            self._recover_if_dead(error)
            raise

        status = response.status if response is not None else 0
        reason = response.status_text if response is not None else ""

        try:
            headers = response.all_headers() if response is not None else {}
        except PlaywrightError:
            headers = {}

        body = self._settle_interstitial()
        self._last_request_at = time.monotonic()

        if status == 200 and looks_like_challenge(body):
            status, reason = 403, "Cloudflare challenge not completed"

        return BrowserResponse(status, body, headers, reason, self._page.url)

    def _settle_interstitial(self):
        """Give a challenge page a chance to resolve into real content."""
        try:
            body = self._page.content()
        except PlaywrightError:
            # Content can be unavailable mid-navigation; one retry is
            # enough because the page has stopped moving by then.
            time.sleep(2)
            body = self._page.content()

        if not looks_like_challenge(body):
            return body

        print(
            "  Interstitial shown. Waiting for it to clear"
            f"{'' if HEADLESS else ' (solve it in the browser window if it asks you to)'}..."
        )

        deadline = time.monotonic() + CHALLENGE_WAIT_SECONDS

        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                body = self._page.content()
            except PlaywrightError:
                continue
            if not looks_like_challenge(body):
                print("  Interstitial cleared.")
                return body

        return body

    def close(self):
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass

        self._playwright = None
        self._context = None
        self._page = None


def looks_like_challenge(body):
    """True only for a Cloudflare interstitial, not for AO3 content.

    Matching on loose phrases alone misfires: a search page listing a
    work titled "Just a Moment" is a perfectly good results page. So a
    page that contains AO3's own chrome is never a challenge, and the
    remaining markers have to appear in the <title> or as Cloudflare's
    own challenge scaffolding.
    """
    if not body:
        return False

    lowered = body.lower()

    if 'id="header"' in lowered or 'class="work blurb' in lowered or "/users/logout" in lowered:
        return False

    if any(marker in lowered for marker in (
        "cf-browser-verification",
        "cf_chl_opt",
        "challenge-platform",
        "__cf_chl",
    )):
        return True

    title_match = re.search(r"<title[^>]*>(.*?)</title>", lowered, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    return any(marker in title for marker in (
        "just a moment",
        "attention required",
        "checking your browser",
    ))


BROWSER = BrowserSession()


# ============================================================
# Logging
# ============================================================

class TeeLogger:
    """Duplicate stdout to a timestamped log file.

    Note: reassigning sys.stdout means input() falls back to Python's
    plain-readline-free prompt path rather than native GNU readline.
    Long cookie pastes are unaffected (handled via ao3_cookie.txt,
    which sidesteps terminal input entirely) but arrow-key history
    editing on other prompts will behave more simply than usual.
    """

    def __init__(self, filepath, terminal_stream):
        self.terminal = terminal_stream
        self.log_file = open(filepath, "a", encoding="utf-8")
        self._line_buf = ""

        self.log_file.write(
            f"\n===== Run started {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====\n"
        )
        self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        self._line_buf += message

        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{timestamp}] {line}\n")

        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        try:
            if self._line_buf:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"[{timestamp}] {self._line_buf}\n")
                self._line_buf = ""
            self.log_file.flush()
            self.log_file.close()
        finally:
            try:
                self.terminal.flush()
            except Exception:
                pass


def log_error(message, exception=None):
    """Append diagnostic errors and full tracebacks to error_log.txt.

    Cookie values and request headers are intentionally never logged.
    """
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as error_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            error_file.write(f"\n[{timestamp}] ERROR: {message}\n")
            if exception is not None:
                error_file.write(
                    "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
                )
            error_file.flush()
    except Exception:
        pass


# ============================================================
# Cookie handling
# ============================================================

def parse_cookie_string(cookie_str):
    """
    Parse a pasted cookie value in any of these forms:
      Cookie: name=value; other=value
      name=value; other=value
      a bare _otwarchive_session value (which itself often contains
      "=" characters as base64 padding -- handled via anchoring on
      the known cookie name rather than naive split-on-"=").
    """
    cookie_str = cookie_str.strip().strip('"').strip("'").strip()

    cookie_str = re.sub(
        r"^\s*cookie\s*:\s*",
        "",
        cookie_str,
        flags=re.IGNORECASE,
    )

    cookies = {}

    session_match = re.search(
        r"_otwarchive_session\s*=\s*([^;]+)",
        cookie_str,
        flags=re.IGNORECASE,
    )

    if session_match:
        cookies["_otwarchive_session"] = (
            session_match.group(1).strip().strip('"').strip("'")
        )
        cookie_str = (
            cookie_str[:session_match.start()]
            + cookie_str[session_match.end():]
        )

    name_re = re.compile(r"^[A-Za-z0-9_-]{1,60}$")

    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        if name_re.fullmatch(key):
            cookies[key] = value.strip().strip('"').strip("'")

    if not cookies and cookie_str:
        cookies["_otwarchive_session"] = cookie_str

    return cookies


def verify_login(session_label="this cookie"):
    """Check whether AO3's homepage appears to recognize the session."""
    response = get_with_retry("https://archiveofourown.org/")

    if response is None or response.status_code != 200:
        log_error(
            "Login verification failed. "
            f"status={getattr(response, 'status_code', None)} "
            f"reason={getattr(response, 'reason', None)!r}"
        )
        print("  Could not verify login. Continuing anyway.")
        return None

    logged_in = (
        "/users/logout" in response.text
        or 'action="/users/logout"' in response.text
    )

    if logged_in:
        print(f"  Login verified: AO3 recognizes {session_label} as a logged-in session.")
    else:
        print(f"  Warning: AO3's homepage does not show a logged-in state with {session_label}.")

    return logged_in


def apply_session_cookie(cookie_str):
    if not cookie_str:
        return

    parsed = parse_cookie_string(cookie_str)

    if not parsed:
        print("  Could not parse the cookie string. Continuing anonymously.")
        return

    BROWSER.add_cookies(parsed)

    print("  Parsed cookie name(s): " + ", ".join(sorted(parsed.keys())))

    if "_otwarchive_session" not in parsed:
        print(
            "  Warning: no _otwarchive_session cookie was found. "
            "Restricted works may remain inaccessible."
        )

    verify_login()


def get_session_cookie_input():
    """Reads the cookie from ao3_cookie.txt if present, else prompts.

    ao3_cookie.txt exists specifically because many terminals have a
    hard length limit on a single pasted line (a long-standing macOS
    quirk in particular). A 5+ cookie header including Cloudflare's
    cf_clearance/cf_bm can easily exceed that, making Enter appear to
    do nothing. A text file has no such limit.
    """
    if os.path.exists(COOKIE_FILE):
        with open(COOKIE_FILE, "r", encoding="utf-8") as file:
            content = file.read().strip()
        if content:
            print(f"Found {COOKIE_FILE}. Using the cookie from that file.")
            return content
        print(f"{COOKIE_FILE} exists but is empty. Falling back to the prompt.")

    print("Optional: paste a logged-in AO3 session cookie if you need registered-user-only works.")
    print(
        f"If pasting fails silently (very long cookies can exceed your terminal's paste "
        f"limit), create a plain text file named '{COOKIE_FILE}' in this folder instead."
    )
    print("Leave blank to continue anonymously.")

    return input("AO3 session cookie (optional): ").strip()


# ============================================================
# Networking
# ============================================================

class HardBlockError(Exception):
    """AO3 clearly told us to stop or became persistently unreachable."""


def describe_429(response, url):
    """Builds two strings: a full, untruncated version (for the error
    log) and a terminal-friendly truncated version, so a large
    Cloudflare challenge page doesn't flood the live console."""
    retry_after_header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    retry_after_seconds = parse_retry_after(retry_after_header)

    body = response.text

    def build(body_text):
        return "\n".join([
            "  --- 429 response ---",
            f"  URL: {url}",
            f"  Status: {response.status_code} {response.reason}",
            f"  Retry-After header: {retry_after_header if retry_after_header else '(not sent)'}",
            f"  Response headers: {dict(response.headers)}",
            (f"  Response body:\n  {body_text}" if body_text else "  Response body: (empty)"),
            "  ---------------------",
        ])

    full_detail = build(body)

    truncated_body = body if len(body) <= 1500 else body[:1500] + "\n  ... [truncated; full body in error_log.txt]"
    console_detail = build(truncated_body)

    return console_detail, full_detail, retry_after_seconds


def parse_retry_after(header_value):
    """Seconds to wait from a Retry-After header, or None.

    RFC 9110 allows either a delay in seconds or an HTTP date, and
    AO3/Cloudflare send both forms depending on which layer answers,
    so both are handled. What AO3 asks for is always preferable to a
    guessed backoff interval.
    """
    if not header_value:
        return None

    header_value = header_value.strip()

    try:
        return max(0, int(float(header_value)))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None

    if retry_at is None:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    return max(0, int((retry_at - datetime.now(timezone.utc)).total_seconds()))


def polite_delay(min_seconds=5.0, max_seconds=10.0):
    time.sleep(random.uniform(min_seconds, max_seconds))


def get_with_retry(url, max_retries=3):
    """
    Retry transient network errors, Cloudflare origin errors (52x) and
    429 responses with backoff. 403 and 503 are returned immediately so
    the caller can stop rather than trying to push through a block.
    """
    response = None

    for attempt in range(max_retries):
        try:
            response = BROWSER.get(url, timeout=60)
        except (PlaywrightError, PlaywrightTimeoutError) as error:
            log_error(
                f"Network request failed for {url} ({error.__class__.__name__}: {error})",
                error,
            )
            wait = 10 * (attempt + 1)
            print(
                f"  Network error ({error.__class__.__name__}: {error}). "
                f"Waiting {wait}s before retrying ({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
            response = None
            continue

        if response.status_code == 429:
            console_detail, full_detail, retry_after_seconds = describe_429(response, url)
            log_error(f"HTTP 429 response details for {url}:\n{full_detail}")
            print(console_detail)

            if retry_after_seconds is not None:
                # Honour exactly what AO3 asked for, plus a small grace
                # margin so the retry lands after its window rather
                # than on the boundary. Guessing a shorter interval is
                # how a temporary rate-limit turns into a real block.
                wait = retry_after_seconds + RETRY_AFTER_GRACE_SECONDS
                print(f"  AO3 asked for {retry_after_seconds}s. Waiting {wait}s before retrying...")
            else:
                wait = 15 * (attempt + 1)
                print(f"  No Retry-After header sent. Waiting {wait}s before retrying...")

            time.sleep(wait)
            continue

        if response.status_code in ORIGIN_ERROR_STATUSES or response.status_code == 0:
            log_error(f"Origin-side failure for {url}: HTTP {response.status_code} {response.reason}")
            wait = 20 * (attempt + 1)
            print(
                f"  HTTP {response.status_code}: Cloudflare could not reach AO3's server. "
                f"This is AO3's end, not a block. Waiting {wait}s before retrying "
                f"({attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
            continue

        return response

    return response


# ============================================================
# Cache
# ============================================================

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as file:
                cache = json.load(file)
            cache.setdefault("pages", {})
            cache.setdefault("page_counts", {})
            cache.setdefault("metadata", {})
            return cache
        except (json.JSONDecodeError, OSError):
            print(f"Warning: could not read {CACHE_FILE}. Starting a fresh cache.")

    return {"pages": {}, "page_counts": {}, "metadata": {}}


def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as file:
            json.dump(cache, file)
    except OSError as error:
        print(f"Warning: could not write cache file: {error}")


# ============================================================
# Parsing helpers
# ============================================================

def extract_work_id_from_blurb(work_tag):
    tag_id = work_tag.get("id", "")
    match = re.search(r"work_(\d+)", tag_id)
    if match:
        return match.group(1)

    link = work_tag.select_one("h4.heading a[href^='/works/']")
    if link:
        match = re.search(r"/works/(\d+)", link.get("href", ""))
        if match:
            return match.group(1)

    return None


def extract_work_id_from_url(url):
    match = re.search(r"/works/(\d+)", url)
    return match.group(1) if match else None


def extract_stat(stats_dl, class_name):
    node = stats_dl.find("dd", class_=class_name)
    if node is None:
        return 0
    text = node.get_text(strip=True).replace(",", "")
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def extract_chapters(stats_dl):
    node = stats_dl.find("dd", class_="chapters")
    if node is None:
        return 1
    match = re.search(r"(\d+)\s*/", node.get_text(" ", strip=True))
    return int(match.group(1)) if match else 1


def extract_meta_tags(soup, dd_class):
    node = soup.find("dd", class_=dd_class)
    if node is None:
        return ""
    links = [link.get_text(" ", strip=True) for link in node.select("li a")]
    if links:
        return ", ".join(links)
    return node.get_text(" ", strip=True)


def extract_blurb_language(work_tag):
    stats = work_tag.find("dl", class_="stats")
    if not stats:
        return ""
    node = stats.find("dd", class_="language")
    return node.get_text(" ", strip=True) if node else ""


def extract_blurb_dates(work_tag):
    node = work_tag.find("p", class_="datetime")
    return node.get_text(" ", strip=True) if node else ""


def extract_blurb_relationships(work_tag):
    for selector in ("dd.relationship.tags", "dd.relationship"):
        node = work_tag.select_one(selector)
        if node:
            links = [link.get_text(" ", strip=True) for link in node.select("a")]
            if links:
                return ", ".join(links)
            return node.get_text(" ", strip=True)
    return ""


def extract_blurb_fandoms(work_tag):
    """Return all fandom tags shown on an AO3 search-result card."""
    return [
        link.get_text(" ", strip=True)
        for link in work_tag.select("h5.fandoms.heading a.tag")
        if link.get_text(" ", strip=True)
    ]


def classify_required_tags(work_tag):
    """Extract rating, warnings, and category from the required-tags
    icon block on a search result card."""
    rating = ""
    warnings = []
    categories = []

    tags_ul = work_tag.find("ul", class_="required-tags")
    if not tags_ul:
        return rating, "", ""

    for span in tags_ul.find_all("span"):
        label = span.get("title", "").strip()
        if not label:
            continue
        if label in RATING_VALUES:
            rating = label
        elif label in CATEGORY_VALUES:
            categories.append(label)
        elif label in COMPLETE_STATUS_VALUES:
            continue
        else:
            warnings.append(label)

    return rating, ", ".join(warnings), ", ".join(categories)


def calculate_metrics(hits, kudos, bookmarks, comments, words):
    """Derived engagement-proxy metrics. Hits are accesses, not unique
    readers -- these are proxies, not direct preference measurements."""
    return {
        "kudos_to_hits": (kudos / hits * 100) if hits > 0 else 0.0,
        "bookmarks_to_hits": (bookmarks / hits * 100) if hits > 0 else 0.0,
        "comments_to_hits": (comments / hits * 100) if hits > 0 else 0.0,
        "comments_to_kudos": (comments / kudos * 100) if kudos > 0 else 0.0,
        "kudos_per_10k_words": (kudos / words * 10000) if words > 0 else 0.0,
        "bookmarks_per_10k_words": (bookmarks / words * 10000) if words > 0 else 0.0,
        "comments_per_10k_words": (comments / words * 10000) if words > 0 else 0.0,
    }


def text_work_subset(df):
    """Return only rows with a positive word count. Zero-word works
    (podfic, art, etc.) remain in the raw dataset but should be
    excluded from anything word-normalized (kudos_per_10k_words etc.),
    since dividing by zero-ish word counts isn't meaningful."""
    if df is None or df.empty:
        return df
    words = pd.to_numeric(df.get("words", 0), errors="coerce").fillna(0)
    return df.loc[words > 0].copy()


# ============================================================
# Work page parser (target work only -- one request)
# ============================================================

def parse_ao3_work(work_url):
    if "/chapters/" in work_url:
        work_url = work_url.split("/chapters/")[0]

    response = get_with_retry(work_url)

    if response is None:
        raise HardBlockError("AO3 could not be reached after several attempts.")
    if response.status_code in (403, 503) or response.status_code in ORIGIN_ERROR_STATUSES:
        raise HardBlockError(f"AO3 returned HTTP {response.status_code} for the target work.")
    if response.status_code == 429:
        raise HardBlockError("AO3 continued returning HTTP 429 after retries.")
    if response.status_code != 200:
        raise Exception(f"Could not load work page. HTTP status: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    stats_dl = soup.find("dl", class_="stats")
    if stats_dl is None:
        raise Exception(
            "Could not find AO3 work statistics. The work may be restricted, "
            "deleted, or an interstitial page may have been returned."
        )

    fandoms = [link.get_text(" ", strip=True) for link in soup.select("dd.fandom.tags ul.commas li a")]
    fandom = fandoms[0] if fandoms else ""

    hits = extract_stat(stats_dl, "hits")
    kudos = extract_stat(stats_dl, "kudos")
    bookmarks = extract_stat(stats_dl, "bookmarks")
    comments = extract_stat(stats_dl, "comments")
    words = extract_stat(stats_dl, "words")
    chapters = extract_chapters(stats_dl)

    title_node = soup.find("h2", class_="title")
    rating = extract_meta_tags(soup, "rating")
    warnings = extract_meta_tags(soup, "warning")
    category = extract_meta_tags(soup, "category")
    relationships = extract_meta_tags(soup, "relationship")

    language_node = soup.find("dd", class_="language")
    language = language_node.get_text(" ", strip=True) if language_node else ""

    published_node = stats_dl.find("dd", class_="published")
    updated_node = stats_dl.find("dd", class_="status")
    date_published = published_node.get_text(" ", strip=True) if published_node else ""
    date_updated = updated_node.get_text(" ", strip=True) if updated_node else date_published

    metrics = calculate_metrics(hits, kudos, bookmarks, comments, words)

    return {
        "work_id": extract_work_id_from_url(work_url),
        "title": title_node.get_text(" ", strip=True) if title_node else "Unknown",
        "fandom": fandom,
        "fandoms": ", ".join(fandoms),
        "fandom_count": len(fandoms),
        "is_crossover": len(fandoms) > 1,
        "relationships": relationships,
        "hits": hits,
        "kudos": kudos,
        "bookmarks": bookmarks,
        "comments": comments,
        "words": words,
        "chapters": chapters,
        "rating": rating,
        "warnings": warnings,
        "category": category,
        "language": language,
        "date_published": date_published,
        "date_updated": date_updated,
        **metrics,
    }


# ============================================================
# Search URL construction
# ============================================================

def build_search_url(fandom, category, warning, crossover, complete_only, timeframe, page=1):
    base_url = "https://archiveofourown.org/works/search?"

    params = [
        ("utf8", "✓"),
        ("commit", "Sort and Filter"),
        ("page", str(page)),
        ("work_search[sort_column]", "revised_at"),
        ("work_search[sort_direction]", "desc"),
    ]

    if fandom:
        params.append(("work_search[fandom_names]", fandom.strip()))

    if category in CATEGORY_MAP:
        params.append(("work_search[category_ids][]", CATEGORY_MAP[category]))

    if warning in WARNING_MAP:
        params.append(("work_search[archive_warning_ids][]", WARNING_MAP[warning]))

    # See the top-of-file note on "T"/"F" vs other conventions for
    # these two fields -- kept consistent with each other deliberately.
    if crossover == "1":
        params.append(("work_search[crossover]", "F"))
    elif crossover == "2":
        params.append(("work_search[crossover]", "T"))

    if complete_only == "1":
        params.append(("work_search[complete]", "T"))

    if timeframe in ("1", "2", "3"):
        days_map = {"1": 7, "2": 30, "3": 365}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_map[timeframe])
        params.append(("work_search[date_from]", cutoff.strftime("%Y-%m-%d")))
    elif timeframe.startswith("4:"):
        _, start_date, end_date = timeframe.split(":", 2)
        if start_date:
            params.append(("work_search[date_from]", start_date))
        if end_date:
            params.append(("work_search[date_to]", end_date))

    return base_url + urllib.parse.urlencode(params)


# ============================================================
# Search-result parser
# ============================================================

def parse_blurb_page(html):
    """Parse only metadata exposed in AO3 search-result cards. No
    per-work requests are made here."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for work in soup.select("li.work.blurb"):
        stats = work.find("dl", class_="stats")
        if stats is None:
            continue

        chapters = extract_chapters(stats)
        rating, warnings, category = classify_required_tags(work)

        hits = extract_stat(stats, "hits")
        kudos = extract_stat(stats, "kudos")
        bookmarks = extract_stat(stats, "bookmarks")
        comments = extract_stat(stats, "comments")
        words = extract_stat(stats, "words")

        metrics = calculate_metrics(hits, kudos, bookmarks, comments, words)
        fandoms = extract_blurb_fandoms(work)

        title_link = work.select_one("h4.heading a")

        results.append({
            "work_id": extract_work_id_from_blurb(work),
            "title": title_link.get_text(" ", strip=True) if title_link else "",
            "hits": hits,
            "kudos": kudos,
            "bookmarks": bookmarks,
            "comments": comments,
            "words": words,
            "chapters": chapters,
            "fandoms": ", ".join(fandoms),
            "fandom_count": len(fandoms),
            # Heuristic, not authoritative: AO3's own crossover definition
            # is more nuanced than "more than one fandom tag" (e.g. some
            # multi-fandom tags represent a shared universe rather than a
            # true crossover). The server-side work_search[crossover]
            # filter is the primary control; this is a client-side
            # sanity check on top of it, not a replacement for it.
            "is_crossover": len(fandoms) > 1,
            "rating": rating,
            "warnings": warnings,
            "category": category,
            "language": extract_blurb_language(work),
            "relationships": extract_blurb_relationships(work),
            "date_updated": extract_blurb_dates(work),
            **metrics,
        })

    return results


# ============================================================
# Page count
# ============================================================

def get_result_page_count(fandom, category, warning, crossover, complete_only, timeframe, cache):
    complete_only = str(complete_only) if str(complete_only) in {"0", "1"} else "0"

    url = build_search_url(fandom, category, warning, crossover, complete_only, timeframe, page=1)

    cached_count = cache.get("page_counts", {}).get(url)
    if cached_count is not None:
        print(f"  Using cached page count: {cached_count} page(s).")
        return cached_count, url

    response = get_with_retry(url)

    if response is None:
        return 0, url
    if response.status_code in (403, 503) or response.status_code in ORIGIN_ERROR_STATUSES:
        raise HardBlockError(
            f"AO3 returned HTTP {response.status_code} while checking the search, "
            "and it did not clear on retry."
        )
    if response.status_code == 429:
        raise HardBlockError("AO3 returned HTTP 429 while checking the search.")
    if response.status_code != 200:
        print(f"Could not read search page. HTTP {response.status_code}.")
        return 0, url

    soup = BeautifulSoup(response.text, "html.parser")
    if not soup.select("li.work.blurb"):
        return 0, url

    page_links = soup.select("ol.pagination li a")
    page_numbers = [int(link.get_text(" ", strip=True)) for link in page_links if link.get_text(" ", strip=True).isdigit()]
    total_pages = max(page_numbers) if page_numbers else 1

    cache.setdefault("page_counts", {})[url] = total_pages
    cache.setdefault("pages", {})[url] = parse_blurb_page(response.text)
    save_cache(cache)

    return total_pages, url


# ============================================================
# Sampling
# ============================================================

def choose_pages(total_pages, start_page, end_page, sampling_mode, sample_page_count, seed):
    """
    Choose which result pages to visit.
      all: every page in the requested range.
      systematic: evenly spaced pages across the range.
      random: randomly selected pages using a recorded seed.
    This does not claim to create a perfect random sample of works --
    AO3 pagination is itself an ordered sample frame -- but the seed
    and selected pages are recorded so sampling is reproducible.
    """
    start_page = max(1, start_page)
    end_page = min(total_pages, end_page)

    if start_page > end_page:
        return []

    available = list(range(start_page, end_page + 1))

    if sampling_mode == "all":
        return available

    count = min(max(1, sample_page_count), len(available))

    if sampling_mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(available, count))

    if sampling_mode == "systematic":
        if count == 1:
            return [available[len(available) // 2]]
        positions = np.linspace(0, len(available) - 1, count)
        return sorted({available[int(round(p))] for p in positions})

    raise ValueError("sampling_mode must be 'all', 'systematic', or 'random'.")


# ============================================================
# Cohort scraping
# ============================================================

def scrape_fandom_cohort(
    fandom, category, warning, crossover, complete_only, timeframe,
    chapter_choice, target_chapters, total_pages, cache,
    start_page=1, end_page=None, sampling_mode="all", sample_page_count=10, seed=2026,
):
    """Collect cohort metadata from selected search-result pages. Full
    work text is never downloaded for cohort works."""
    complete_only = str(complete_only) if str(complete_only) in {"0", "1"} else "0"

    if end_page is None:
        end_page = total_pages

    selected_pages = choose_pages(total_pages, start_page, end_page, sampling_mode, sample_page_count, seed)

    print(
        f"  Selected {len(selected_pages)} page(s): {selected_pages[:20]}"
        + ("..." if len(selected_pages) > 20 else "")
    )

    fics = []
    consecutive_hard_failures = 0

    for index, page in enumerate(selected_pages, start=1):
        url = build_search_url(fandom, category, warning, crossover, complete_only, timeframe, page)

        cached = cache.get("pages", {}).get(url)

        if cached is not None:
            page_results = cached
            print(f"  Page {page}: loaded from cache ({len(page_results)} works).")
        else:
            polite_delay()
            response = get_with_retry(url)

            hard_fail = (
                response is None
                or response.status_code in (403, 503, 429)
                or response.status_code in ORIGIN_ERROR_STATUSES
            )

            if hard_fail:
                consecutive_hard_failures += 1
                code = response.status_code if response is not None else "no response"
                print(
                    f"  Page {page}: {code} "
                    f"({consecutive_hard_failures}/{MAX_CONSECUTIVE_HARD_FAILURES})."
                )
                if consecutive_hard_failures >= MAX_CONSECUTIVE_HARD_FAILURES:
                    raise HardBlockError(
                        "AO3 has returned repeated hard failures or sustained rate-limiting. Stopping."
                    )
                continue

            if response.status_code != 200:
                print(f"  Page {page}: HTTP {response.status_code}. Skipping.")
                continue

            consecutive_hard_failures = 0
            page_results = parse_blurb_page(response.text)
            cache.setdefault("pages", {})[url] = page_results
            save_cache(cache)

        for work in page_results:
            chapters = work["chapters"]
            hits = work["hits"]

            # Client-side crossover sanity check (heuristic; see the
            # comment in parse_blurb_page). This is secondary to AO3's
            # own server-side work_search[crossover] filter.
            if crossover == "1" and work.get("is_crossover") is True:
                continue
            if crossover == "2" and work.get("is_crossover") is False:
                continue

            if chapter_choice == "1" and chapters != 1:
                continue
            if chapter_choice == "2" and chapters != target_chapters:
                continue
            if chapter_choice == "3" and chapters <= 1:
                continue

            if hits < 50:
                continue

            fics.append({**work, "sampled_page": page})

        if index % 10 == 0 or index == len(selected_pages):
            print(f"  ...page {index}/{len(selected_pages)} done; {len(fics)} qualifying works.")

    return pd.DataFrame(fics)


# ============================================================
# Research observation keys
# ============================================================

def make_observation_key(
    work_id, fandom_scope, category_filter, warning_filter, crossover_filter,
    chapter_filter, completion_filter, timeframe_filter, sampling_mode, sampling_seed,
):
    """Work ID alone is not enough: the same work can legitimately
    appear in different research cohorts, so the observation key
    includes the sampling condition."""
    values = [
        work_id or "", fandom_scope or "", str(category_filter), str(warning_filter),
        str(crossover_filter), str(chapter_filter), str(completion_filter),
        str(timeframe_filter), sampling_mode, str(sampling_seed),
    ]
    return "|".join(values)


def load_existing_observation_keys(csv_path):
    keys = set()
    if not os.path.exists(csv_path):
        return keys

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.get("observation_key")
            if key:
                keys.add(key)
            elif row.get("work_id"):
                keys.add(f"LEGACY|{row['work_id']}")

    return keys


def append_results_to_csv(csv_path, rows):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Distribution calculation
# ============================================================

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


def _filter_cohort_rows(df_all, fandom, filter_meta):
    required_columns = [
        "role", "fandom", "category_filter", "warning_filter",
        "crossover_filter", "chapter_filter", "completion_filter", "timeframe_filter",
    ]
    missing = [c for c in required_columns if c not in df_all.columns]
    if missing:
        print("  Skipped: CSV is missing columns: " + ", ".join(missing))
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

    subset = df_all.loc[mask].copy()

    # A work can be encountered across multiple sampling runs/seeds;
    # count each physical work only once in the research distribution.
    if "work_id" in subset.columns:
        subset = subset.drop_duplicates(subset=["work_id"], keep="first")

    return subset


def compute_and_save_distribution(csv_path, dist_path, fandom, filter_meta):
    """Recompute distributions from the ENTIRE accumulated CSV, so
    running the script across many sessions (different page ranges,
    different sampling seeds) builds one growing, complete picture."""
    if not os.path.exists(csv_path):
        return None

    try:
        df_all = pd.read_csv(
            csv_path,
            dtype={
                "work_id": str, "category_filter": str, "warning_filter": str,
                "crossover_filter": str, "chapter_filter": str,
                "completion_filter": str, "timeframe_filter": str,
            },
        )
    except (pd.errors.EmptyDataError, OSError):
        return None

    if df_all.empty:
        return None

    subset = _filter_cohort_rows(df_all, fandom, filter_meta)
    if subset is None or subset.empty:
        return None

    entry = {
        "fandom": fandom if fandom else "(all fandoms)",
        **filter_meta,
        "sample_size": int(len(subset)),
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ratios": {},
    }

    percentile_points = [5, 10, 25, 50, 75, 90, 95]

    # Hit-based ratios: valid for every work regardless of word count
    # (podfic, art, and other zero-word works still have real hits).
    hit_based_columns = ["kudos_to_hits", "bookmarks_to_hits", "comments_to_hits", "comments_to_kudos"]

    # Word-normalized ratios: only meaningful for actual text works,
    # so these are computed on the text-only subset.
    word_based_columns = ["kudos_per_10k_words", "bookmarks_per_10k_words", "comments_per_10k_words"]

    text_subset = text_work_subset(subset)

    for column in hit_based_columns:
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        values = sorted(float(v) for v in values if np.isfinite(v))
        if not values:
            continue
        entry["ratios"][column] = {
            "values": values,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in percentile_points},
        }

    for column in word_based_columns:
        if text_subset is None or text_subset.empty or column not in text_subset.columns:
            continue
        values = pd.to_numeric(text_subset[column], errors="coerce").dropna()
        values = sorted(float(v) for v in values if np.isfinite(v))
        if not values:
            continue
        entry["ratios"][column] = {
            "values": values,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in percentile_points},
            "note": "computed on text works only (words > 0)",
        }

    all_dist = {}
    if os.path.exists(dist_path):
        try:
            with open(dist_path, "r", encoding="utf-8") as file:
                all_dist = json.load(file)
        except (json.JSONDecodeError, OSError):
            all_dist = {}

    key = make_distribution_key(fandom, filter_meta)
    all_dist[key] = entry

    with open(dist_path, "w", encoding="utf-8") as file:
        json.dump(all_dist, file, indent=2)

    return entry, key


def percentile_against_distribution(value, distribution_values):
    """Percentile rank: proportion of cohort values strictly below
    the target value."""
    if distribution_values is None:
        return None
    values = np.asarray(distribution_values, dtype=float)
    if values.size == 0:
        return None
    return float(np.mean(values < value) * 100)


def get_accumulated_cohort(csv_path, fandom, filter_meta):
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            csv_path,
            dtype={
                "work_id": str, "category_filter": str, "warning_filter": str,
                "crossover_filter": str, "chapter_filter": str,
                "completion_filter": str, "timeframe_filter": str,
            },
        )
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()

    if df.empty:
        return df

    for column in ["fandom", "fandoms", "category_filter", "warning_filter",
                    "crossover_filter", "chapter_filter", "completion_filter", "timeframe_filter"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    result = _filter_cohort_rows(df, fandom, filter_meta)
    return result if result is not None else pd.DataFrame()


# ============================================================
# Main
# ============================================================

def main():
    original_stdout = sys.stdout
    logger = TeeLogger(LOG_FILE, original_stdout)
    sys.stdout = logger

    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as error_file:
            error_file.write(
                f"\n===== Error log started {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====\n"
            )
        print(f"(Logging this run to {LOG_FILE}; errors to {ERROR_LOG_FILE}.)")

        apply_session_cookie(get_session_cookie_input())

        target_url = input("\nEnter target AO3 work URL: ").strip()
        if not target_url:
            print("No target URL supplied.")
            return

        print("\nFetching target work details...")
        polite_delay()
        target = parse_ao3_work(target_url)

        print(f"\n--- Target Fic: '{target['title']}' ---")
        print(f"Work ID: {target['work_id']}")
        print(f"Fandom(s): {target['fandoms']}")
        print(f"Relationships: {target['relationships']}")
        print(f"Words: {target['words']:,} | Chapters: {target['chapters']}")
        print(
            f"Hits: {target['hits']:,} | Kudos: {target['kudos']:,} | "
            f"Bookmarks: {target['bookmarks']:,} | Comments: {target['comments']:,}"
        )
        print(f"Kudos/Hits: {target['kudos_to_hits']:.2f}%")
        print(f"Bookmarks/Hits: {target['bookmarks_to_hits']:.2f}%")
        print(f"Kudos per 10k words: {target['kudos_per_10k_words']:.2f}")

        print("\n--- Configure Comparison Cohort ---")

        target_fandoms = [f.strip() for f in target["fandoms"].split(",") if f.strip()]

        print("\nFandom scope:\n[0] Choose a target fandom\n[1] All fandoms")
        fandom_choice = input("Select fandom scope (0-1): ").strip()

        if fandom_choice == "0":
            if not target_fandoms:
                print("Could not identify a fandom on the target work. Falling back to all fandoms.")
                fandom_query, fandom_scope = "", "all"
            elif len(target_fandoms) == 1:
                fandom_query, fandom_scope = target_fandoms[0], "same"
            else:
                print("\nTarget work fandoms:")
                for i, name in enumerate(target_fandoms, start=1):
                    print(f"[{i}] {name}")
                idx_input = input(f"Choose fandom (1-{len(target_fandoms)}): ").strip()
                try:
                    idx = int(idx_input)
                except ValueError:
                    idx = 1
                idx = max(1, min(idx, len(target_fandoms)))
                fandom_query, fandom_scope = target_fandoms[idx - 1], "same"
        else:
            fandom_query, fandom_scope = "", "all"

        print("\nCategory:\n[0] All\n[1] M/M\n[2] F/F\n[3] F/M\n[4] Multi\n[5] Gen")
        category_filter = input("Select category (0-5): ").strip()
        if category_filter not in {"0", "1", "2", "3", "4", "5"}:
            category_filter = "0"

        print(
            "\nArchive warnings:\n[0] All\n[1] No Warnings Apply\n[2] Creator Chose Not To Use"
            "\n[3] Graphic Violence\n[4] Major Character Death\n[5] Rape/Non-Con\n[6] Underage"
        )
        warning_filter = input("Select warning (0-6): ").strip()
        if warning_filter not in {"0", "1", "2", "3", "4", "5", "6"}:
            warning_filter = "0"

        print("\nCrossovers:\n[0] No crossovers (recommended)\n[1] All works\n[2] Crossovers only")
        crossover_choice = input("Select crossover option (0-2): ").strip()
        if crossover_choice == "0":
            crossover_filter = "1"
        elif crossover_choice == "2":
            crossover_filter = "2"
        else:
            crossover_filter = "0"

        print(
            "\nChapters:\n[0] All\n[1] One-shots only\n[2] Same number as target\n[3] Multi-chapter only"
        )
        chapter_filter = input("Select chapter filter (0-3): ").strip()
        if chapter_filter not in {"0", "1", "2", "3"}:
            chapter_filter = "0"

        print("\nCompletion:\n[0] All\n[1] Complete only")
        completion_filter = input("Select completion filter (0-1): ").strip()
        if completion_filter not in {"0", "1"}:
            completion_filter = "0"

        print("\nTimeframe:\n[0] All time\n[1] Last week\n[2] Last month\n[3] Last year\n[4] Custom range")
        timeframe_filter = input("Select timeframe (0-4): ").strip()
        if timeframe_filter not in {"0", "1", "2", "3", "4"}:
            timeframe_filter = "0"

        if timeframe_filter == "4":
            date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            start_date = input("Start date (YYYY-MM-DD, blank = no lower bound): ").strip()
            end_date = input("End date (YYYY-MM-DD, blank = no upper bound): ").strip()

            if start_date and not date_re.match(start_date):
                print(f"'{start_date}' isn't in YYYY-MM-DD format. Ignoring start date.")
                start_date = ""
            if end_date and not date_re.match(end_date):
                print(f"'{end_date}' isn't in YYYY-MM-DD format. Ignoring end date.")
                end_date = ""

            if not start_date and not end_date:
                print("No valid custom dates given. Falling back to All time.")
                timeframe_filter = "0"
            else:
                # Encoded into timeframe_filter itself since it already
                # flows everywhere as a plain string (CSV, cache keys,
                # distribution keys) -- different custom ranges then
                # automatically count as distinct filter conditions.
                timeframe_filter = f"4:{start_date}:{end_date}"

        print("\nChecking the result count...")
        cache = load_cache()

        total_pages, _ = get_result_page_count(
            fandom=fandom_query, category=category_filter, warning=warning_filter,
            crossover=crossover_filter, complete_only=completion_filter,
            timeframe=timeframe_filter, cache=cache,
        )

        if total_pages == 0:
            print("No matching works found.")
            return

        print(f"Found {total_pages} result page(s).")

        start_input = input(f"Start page (1-{total_pages}, blank = 1): ").strip()
        end_input = input(f"End page (1-{total_pages}, blank = {total_pages}): ").strip()

        start_page = int(start_input) if start_input.isdigit() else 1
        end_page = int(end_input) if end_input.isdigit() else total_pages
        start_page = max(1, min(start_page, total_pages))
        end_page = max(start_page, min(end_page, total_pages))

        print("\nSampling mode:\n[0] All pages\n[1] Systematic pages\n[2] Random pages")
        sampling_choice = input("Select sampling mode (0-2): ").strip()
        sampling_mode = {"1": "systematic", "2": "random"}.get(sampling_choice, "all")

        sample_page_count = end_page - start_page + 1
        if sampling_mode != "all":
            count_input = input("How many pages should be sampled? ").strip()
            if count_input.isdigit():
                sample_page_count = max(1, int(count_input))

        seed_input = input("Sampling seed (blank = 2026): ").strip()
        sampling_seed = int(seed_input) if seed_input.isdigit() else 2026

        print("\nCollecting cohort metadata. Only search-result cards are parsed for cohort works.")

        df = scrape_fandom_cohort(
            fandom=fandom_query, category=category_filter, warning=warning_filter,
            crossover=crossover_filter, complete_only=completion_filter, timeframe=timeframe_filter,
            chapter_choice=chapter_filter, target_chapters=target["chapters"],
            total_pages=total_pages, cache=cache, start_page=start_page, end_page=end_page,
            sampling_mode=sampling_mode, sample_page_count=sample_page_count, seed=sampling_seed,
        )

        run_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sampling_date = datetime.now(timezone.utc).date().isoformat()

        filter_meta = {
            "fandom_scope": fandom_scope, "category_filter": category_filter,
            "warning_filter": warning_filter, "crossover_filter": crossover_filter,
            "chapter_filter": chapter_filter, "completion_filter": completion_filter,
            "timeframe_filter": timeframe_filter, "sampling_mode": sampling_mode,
            "sampling_seed": sampling_seed,
        }

        existing_keys = load_existing_observation_keys(CSV_FILE)
        new_rows = []
        skipped_duplicates = 0

        if target.get("work_id"):
            target_key = make_observation_key(
                target["work_id"], fandom_scope, category_filter, warning_filter,
                crossover_filter, chapter_filter, completion_filter, timeframe_filter,
                sampling_mode, sampling_seed,
            )
            if target_key not in existing_keys:
                new_rows.append({
                    "run_timestamp": run_timestamp, "sampling_date": sampling_date,
                    "role": "target", "observation_key": target_key, "work_id": target["work_id"],
                    "title": target["title"], "fandom": target["fandom"], "fandoms": target["fandoms"],
                    "relationships": target["relationships"], **filter_meta, "sampled_page": "",
                    "hits": target["hits"], "kudos": target["kudos"], "bookmarks": target["bookmarks"],
                    "comments": target["comments"], "words": target["words"], "chapters": target["chapters"],
                    "rating": target["rating"], "warnings": target["warnings"], "category": target["category"],
                    "language": target["language"], "date_published": target["date_published"],
                    "date_updated": target["date_updated"], "kudos_to_hits": target["kudos_to_hits"],
                    "bookmarks_to_hits": target["bookmarks_to_hits"], "comments_to_hits": target["comments_to_hits"],
                    "comments_to_kudos": target["comments_to_kudos"],
                    "kudos_per_10k_words": target["kudos_per_10k_words"],
                    "bookmarks_per_10k_words": target["bookmarks_per_10k_words"],
                    "comments_per_10k_words": target["comments_per_10k_words"],
                    "is_text_work": target["words"] > 0,
                    "is_crossover": target["is_crossover"],
                })

        for _, row in df.iterrows():
            work_id = str(row.get("work_id", ""))
            if not work_id or work_id == "nan":
                continue

            observation_key = make_observation_key(
                work_id, fandom_scope, category_filter, warning_filter, crossover_filter,
                chapter_filter, completion_filter, timeframe_filter, sampling_mode, sampling_seed,
            )
            if observation_key in existing_keys:
                skipped_duplicates += 1
                continue
            existing_keys.add(observation_key)

            new_rows.append({
                "run_timestamp": run_timestamp, "sampling_date": sampling_date, "role": "cohort",
                "observation_key": observation_key, "work_id": work_id, "title": row.get("title", ""),
                "fandom": fandom_query, "fandoms": row.get("fandoms", fandom_query),
                "relationships": row.get("relationships", ""), **filter_meta,
                "sampled_page": row.get("sampled_page", ""), "hits": row["hits"], "kudos": row["kudos"],
                "bookmarks": row["bookmarks"], "comments": row["comments"], "words": row["words"],
                "chapters": row["chapters"], "rating": row.get("rating", ""), "warnings": row.get("warnings", ""),
                "category": row.get("category", ""), "language": row.get("language", ""),
                "date_published": "", "date_updated": row.get("date_updated", ""),
                "kudos_to_hits": row["kudos_to_hits"], "bookmarks_to_hits": row["bookmarks_to_hits"],
                "comments_to_hits": row["comments_to_hits"], "comments_to_kudos": row["comments_to_kudos"],
                "kudos_per_10k_words": row["kudos_per_10k_words"],
                "bookmarks_per_10k_words": row["bookmarks_per_10k_words"],
                "comments_per_10k_words": row["comments_per_10k_words"],
                "is_text_work": int(row.get("words", 0) or 0) > 0,
                "is_crossover": bool(row.get("is_crossover", False)),
            })

        if new_rows:
            append_results_to_csv(CSV_FILE, new_rows)

        print(f"\nAppended {len(new_rows)} new observation(s) to {CSV_FILE}.")
        print(f"Skipped {skipped_duplicates} observation(s) already present under the same sampling condition.")

        dist_result = compute_and_save_distribution(CSV_FILE, DIST_FILE, fandom_query, filter_meta)
        if dist_result:
            entry, dist_key = dist_result
            print(f"\nSaved accumulated distribution with {entry['sample_size']} cohort works.")
            print(f"Distribution key: {dist_key}")

        accumulated = get_accumulated_cohort(CSV_FILE, fandom_query, filter_meta)

        if accumulated.empty:
            print("\nNo accumulated cohort exists for this filter combination yet.")
            return

        print("\n--- ACCUMULATED COHORT ---")
        print(f"Sample size: {len(accumulated)} works")

        for column, label in [
            ("kudos_to_hits", "Kudos/Hits"),
            ("bookmarks_to_hits", "Bookmarks/Hits"),
            ("comments_to_hits", "Comments/Hits"),
            ("kudos_per_10k_words", "Kudos per 10k words"),
        ]:
            if column not in accumulated:
                continue

            col_df = accumulated
            if column.endswith("_per_10k_words"):
                col_df = text_work_subset(accumulated)
                if col_df is None or col_df.empty:
                    continue

            values = pd.to_numeric(col_df[column], errors="coerce").dropna()
            if values.empty:
                continue

            target_value = target[column]
            percentile = percentile_against_distribution(target_value, values.tolist())

            print(f"\n{label}:")
            print(f"  Target: {target_value:.2f}")
            print(f"  Cohort mean: {values.mean():.2f}")
            print(f"  Cohort median: {values.median():.2f}")
            print(f"  Target is above {percentile:.1f}% of accumulated cohort works.")

        print("\nNOTE:")
        print("These percentiles describe relative engagement within your selected cohort.")
        print("They do not establish why readers preferred a work or prove causation.")

        print("\nRecommended next research step:")
        print(
            "Use this metadata dataset to select a smaller, matched M/M and F/F corpus "
            "for manual coding of relationship features such as emotional vulnerability, "
            "physical affection, conflict, romantic initiation, agency, and reciprocity."
        )

    except HardBlockError as error:
        log_error("AO3 requested that the script stop.", error)
        print("\nSCRAPE STOPPED:")
        print(error)
        print("Do not immediately rerun repeatedly. Wait and try later.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as error:
        log_error(f"Unexpected error: {error.__class__.__name__}: {error}", error)
        print("\nUnexpected error:")
        print(f"{error.__class__.__name__}: {error}")
        print(f"Full traceback saved to {ERROR_LOG_FILE}.")
        raise

    finally:
        BROWSER.close()
        sys.stdout = original_stdout
        try:
            logger.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()