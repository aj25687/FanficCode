#!/usr/bin/env python3
"""
ao3_fix_sample.py
=================
Finds the parts of ao3_testing_sample.csv that sample_checker.py flags with
"!" or "!!!", then repairs the TWO WORST (fandom, dimension) pairs by swapping
some works out of the over-represented tag and replacing them with works drawn
RANDOMLY from the live AO3 database. Repeats in rounds until nothing is flagged
(or it runs out of budget / stops making progress).

THE PROBLEM THIS SOLVES (why not just "page 3,417 of the search"?)
------------------------------------------------------------------
AO3 only serves the first 5,000 result pages (5,000 x 20 = 100,000 works) of any
search. Harry Potter has ~718,000 works, so a plain search can never reach ~86%
of them, and "random page" would only ever sample the newest/oldest 100k.

HOW A GENUINELY UNIFORM DRAW IS DONE INSTEAD
--------------------------------------------
1. SLICE the target population into date windows (work_search[date_from/date_to],
   i.e. last-updated date) that each hold <= ~98,000 works, so every work in every
   window is reachable. Windows are found by counting ("N Works found") per year
   and halving any year that is still too big. They do not overlap and leave no
   gaps; the script checks that the window counts add up to the unsliced total
   and refuses to continue if they do not. The date-boundary convention
   (is date_to inclusive?) is detected automatically with a few count queries.
2. DRAW r = a uniform random integer in [0, total works in the stratum), using the
   operating system's entropy (random.SystemRandom), not a seeded generator,
   unless you pass --seed.
3. Walk the window counts cumulatively to find which window r falls in and the
   offset inside it. page = offset // 20 + 1, position = offset % 20.
   Fetch exactly that page and take exactly that work.
   Every work in the stratum therefore has the same chance: 1/total.
4. A draw is only re-done (never "nudged") when the work is already in the
   sample / was just removed, is not first-listed in the fandom, or is
   ineligible. Redrawing from the remainder keeps the draw uniform over what is
   left. Candidate hits / kudos / bookmarks are NEVER used to choose anything.

The STRATUM is "this fandom AND the tag that is under-represented", e.g.
"Arcane + Explicit", so the replacement is random within the tag you need.

WHICH AO3 PAGE IS QUERIED (matches your population numbers)
-----------------------------------------------------------
Population counts come from the filter sidebar of a fandom TAG page, so draws use
that same page and filter style, not the /works/search text form:

  /works?tag_id=Arcane%3A+League+of+Legends+%28Cartoon+2021%29
        &work_search[sort_column]=revised_at&work_search[crossover]=F
        &include_work_search[character_ids][]=2774622
        &work_search[date_from]=...&work_search[date_to]=...&page=N

Rating / warning / category IDs are fixed AO3 constants. Character and
relationship IDs (like 2774622) are READ FROM THAT FANDOM'S OWN SIDEBAR and cached,
so the tag names the checker uses are matched to AO3's real tag IDs, synonyms
included. The fandom tag name is taken from the `fandom` column of your sample CSV.

WHICH WORK GETS REMOVED
-----------------------
Random among the works carrying the over-represented tag, restricted to the ones
whose removal (together with the drawn replacement) actually improves the match.
A swap is accepted only if ALL are true:
  * the targeted dimension's squared deviation shrinks,
  * the fandom's total squared deviation (all 5 dimensions) shrinks,
  * the number of ! / !!! flags in that fandom does not go up.
So fixing one dimension cannot quietly wreck another. The replacement inherits
the removed row's sample_id. Every removal is logged to sample_removed_works.csv.

FILES
-----
  reads / rewrites : ao3_testing_sample.csv   (a timestamped backup is made first)
  writes           : checker_summary.txt      (by sample_checker.py, overwritten)
  appends          : fixer_log.txt, sample_removed_works.csv, ao3_random_strata_cache.json
  needs            : sample_checker.py (same folder), ao3statswithplaywright.py
                     (found automatically, same search as ao3_supply.py)
  optional         : ao3_enrich_sample.py (run at the end to fill the full work-page data)

USAGE
-----
  python3 ao3_fix_sample.py --dry-run     # report flags + plan, NO network, NO changes
  python3 ao3_fix_sample.py --selftest    # ~15 AO3 requests: verifies filter names, date logic, HTML parsing
  python3 ao3_fix_sample.py               # the real run (asks to confirm)
  python3 ao3_fix_sample.py --yes --max-swaps 20 --seed 123
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.parse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

PAGE_SIZE = 20
MAX_PAGES = 5000
SAFE_CAP = 98_000          # largest window we allow (100,000 reachable, minus a margin for drift)
EARLIEST = date(1970, 1, 1)
BASE_URL = "https://archiveofourown.org/works?"

# AO3 tag names used in the fandom filter. The first check the script makes is
# that the live count for each is close to the checker's population total; if a
# name is wrong that check fails loudly instead of sampling the wrong thing.
# The CANONICAL tag each population is counted for (the tag whose /tags/<name>/works page you took the sidebar numbers
# from). Use the full canonical name: a short or non-canonical form can make AO3 redirect and silently drop filters.
# Override one at runtime with:  --fandom-tag "Avatar=Avatar: The Last Airbender (Cartoon 2005)"
FANDOM_AO3_TAG = {
    "Harry Potter": "Harry Potter - J. K. Rowling",
    "Arcane": "Arcane: League of Legends (Cartoon 2021)",
    "Avatar": "Avatar: The Last Airbender (Cartoon 2005)",
}

RATING_IDS = {"Not Rated": "9", "General Audiences": "10", "Teen And Up Audiences": "11",
              "Mature": "12", "Explicit": "13"}
WARNING_IDS = {"Creator Chose Not To Use Archive Warnings": "14", "No Archive Warnings Apply": "16",
               "Graphic Depictions Of Violence": "17", "Major Character Death": "18",
               "Rape/Non-Con": "19", "Underage Sex": "20"}
CATEGORY_IDS = {"Gen": "21", "F/M": "22", "M/M": "23", "Other": "24", "F/F": "116", "Multi": "2246"}

RATING_VALUES = set(RATING_IDS)
CATEGORY_VALUES = set(CATEGORY_IDS)

# CSV field -> include_work_search[<this>][] parameter on the tag works page
FIELD_ID_PARAM = {"rating": "rating_ids", "warnings": "archive_warning_ids", "category": "category_ids",
                  "characters": "character_ids", "relationships": "relationship_ids"}
STATIC_IDS = {"rating": RATING_IDS, "warnings": WARNING_IDS, "category": CATEGORY_IDS}

ENRICH_FIELDS_ORDER = [
    "sample_id", "work_id", "url", "title", "fandom", "rating", "warnings", "category",
    "relationships", "characters", "freeform_tags", "date_published", "date_updated",
    "words", "chapters", "language", "hits", "kudos", "bookmarks", "comments",
    "kudos_to_hits", "bookmarks_to_hits", "comments_to_hits", "comments_to_kudos",
    "kudos_per_10k_words", "bookmarks_per_10k_words", "comments_per_10k_words", "tags_enriched",
]

LOG_FILE = "fixer_log.txt"
REMOVED_FILE = "sample_removed_works.csv"
STRATA_CACHE = "ao3_random_strata_cache.json"

CFG = None  # argparse namespace, set in main()

# Which URL form actually applies filters, per fandom (chosen by StratumStore.ensure_style, verified against the sidebar).
#   tag_id   : /works?tag_id=<tag>&include_work_search[...]
#   tag_path : /tags/<tag>/works?include_work_search[...]
#   search   : /works/search?work_search[fandom_names]=<tag>&work_search[...]
STYLES = ["tag_id", "tag_path", "search"]
FANDOM_STYLE = {}


class BlockedError(Exception):
    """AO3 refused or kept failing. Stop, keep what is saved."""


class SliceError(Exception):
    """The date windows could not be proven to partition the stratum."""


class FilterError(Exception):
    """AO3 is returning works that do not match the filter we sent."""


class TagIdError(Exception):
    """A tag name could not be matched to an AO3 tag ID from the fandom sidebar."""


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

class LogTee:
    """stdout -> terminal + timestamped, appended log file."""

    def __init__(self, path, terminal):
        self.terminal = terminal
        self.fh = open(path, "a", encoding="utf-8")
        self.buf = ""
        self.fh.write(f"\n===== Run started {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====\n")
        self.fh.flush()

    def write(self, text):
        self.terminal.write(text)
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self.fh.write(f"[{stamp}] {line}\n")
        self.fh.flush()

    def flush(self):
        self.terminal.flush()
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# LOADING THE OTHER SCRIPTS
# ══════════════════════════════════════════════════════════════

def _import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_checker(explicit=None):
    candidates = [Path(explicit)] if explicit else []
    candidates += [HERE / "sample_checker.py", Path.cwd() / "sample_checker.py"]
    for p in candidates:
        if p.exists():
            module = _import_path("sample_checker", p)
            _require_compatible_checker(module, p)
            return module
    sys.exit("ERROR: sample_checker.py not found next to this script (use --checker PATH).")


REQUIRED_CHECKER_API = ["apply_population", "FANDOMS", "DIMENSIONS", "read_csv", "detect_fandom", "count_matched_tags",
                        "dimension_stats", "analyze_rows", "flagged_items", "is_flagged", "run_check"]


def _require_compatible_checker(module, path):
    """The fixer needs the refactored sample_checker.py (the one that writes checker_summary.txt).
    An older copy lying in the same folder fails here, with a clear message, instead of mid-run."""
    import inspect
    missing = [n for n in REQUIRED_CHECKER_API if not hasattr(module, n)]
    if not missing and len(inspect.signature(module.run_check).parameters) < 3:
        missing.append("run_check(csv_path, sep, summary_path)")
    if missing:
        sys.exit(f"ERROR: {path} is an older/different sample_checker.py (missing: {', '.join(missing)}).\n"
                 "Replace it with the sample_checker.py delivered alongside this script "
                 "(it writes checker_summary.txt and exposes the functions the fixer calls), then re-run.\n"
                 "Tip: keep your old one under another name if you still need it.")


def load_scraper(explicit=None):
    """Same search as ao3_supply.py: next to this file and up its parents, in V4/ and FanficCode/V4/."""
    name = "ao3statswithplaywright.py"
    candidates = [Path(explicit)] if explicit else []
    for folder in [Path.cwd(), HERE, *HERE.parents][:6]:
        candidates += [folder / name, folder / "V4" / name, folder / "FanficCode" / "V4" / name]
    for p in candidates:
        if p.exists():
            print(f"  Using scraper/browser layer: {p}")
            return _import_path("ao3_scraper_module", p)
    sys.exit(f"ERROR: could not find {name}. Pass --scraper-path /full/path/to/{name}.")


# ══════════════════════════════════════════════════════════════
# NETWORK
# ══════════════════════════════════════════════════════════════

class LiveFetcher:
    """Your existing Playwright browser, pacing (10-15 s between loads) and retry logic."""

    def __init__(self, scraper_path, cookie_file):
        self.scraper = load_scraper(scraper_path)
        if cookie_file and os.path.exists(cookie_file):
            raw = Path(cookie_file).read_text(encoding="utf-8").strip()
            if raw:
                self.scraper.BROWSER.add_cookies(self.scraper.parse_cookie_string(raw))
                print(f"  Loaded cookie(s) from {cookie_file}.")

    last_url = ""

    def get(self, url):
        resp = self.scraper.get_with_retry(url)
        if resp is None:
            return 0, ""
        self.last_url = getattr(resp, "url", "") or ""
        return resp.status_code, resp.text

    def close(self):
        try:
            self.scraper.BROWSER.close()
        except Exception:
            pass


class Client:
    """Wraps a fetcher: counts requests and stops after repeated hard failures."""

    def __init__(self, fetcher, max_failures=3):
        self.fetcher = fetcher
        self.requests = 0
        self.failures = 0
        self.max_failures = max_failures

    def get(self, url):
        self.requests += 1
        status, text = self.fetcher.get(url)
        if status == 200:
            self.failures = 0
            return text
        self.failures += 1
        print(f"  !! HTTP {status} ({self.failures}/{self.max_failures}) for {url[:110]}...")
        if self.failures >= self.max_failures:
            raise BlockedError("AO3 returned repeated failures / blocks. Stopping; progress so far is saved. "
                               "Wait a while before re-running.")
        return None

    def get_required(self, url):
        text = self.get(url)
        while text is None:  # get() raises once the failure budget is spent
            text = self.get(url)
        return text


# ══════════════════════════════════════════════════════════════
# PARSING AO3 SEARCH PAGES
# ══════════════════════════════════════════════════════════════

_COUNT_RE = re.compile(r"([\d,]+)\s+Works?\b", re.IGNORECASE)


def parse_work_count(html):
    """Tag works page: '1 - 20 of 66,102 Works in <tag>' / '5 Works in <tag>'.
    (Also accepts '... Works found in ...'.) 'No results'/'0 Works' -> 0; unreadable -> None."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for heading in soup.select("h2.heading, h3.heading"):
        m = _COUNT_RE.search(heading.get_text(" ", strip=True))
        if m:
            return int(m.group(1).replace(",", ""))
    text = soup.get_text(" ", strip=True)
    if re.search(r"No results found|did not match any works|0 Works", text, re.IGNORECASE):
        return 0
    return None


def encode_tag_path(name):
    """AO3's /tags/<name>/works path escaping."""
    for a, b in (("/", "*s*"), ("&", "*a*"), (".", "*d*"), ("#", "*h*"), ("?", "*q*")):
        name = name.replace(a, b)
    return urllib.parse.quote(name, safe="*:()',-")


def norm_tag(name):
    return re.sub(r"\s+", " ", name or "").strip().lower()


_LABEL_RE = re.compile(r"^(.*?)\s*\(([\d,]+)\)\s*$")


def parse_sidebar(html):
    """Filter sidebar -> {field: {normalised name: {"id", "name", "count"}}} for the five checked fields.
    Reads every include_work_search[<x>_ids][] checkbox and its label text 'Tag name (12,345)'."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    param_to_field = {v: k for k, v in FIELD_ID_PARAM.items()}
    out = {f: {} for f in FIELD_ID_PARAM}
    for inp in soup.select("input[type=checkbox]"):
        m = re.match(r"include_work_search\[(\w+)\]\[\]", inp.get("name", ""))
        if not m or m.group(1) not in param_to_field:
            continue
        label = inp.find_parent("label")
        text = label.get_text(" ", strip=True) if label else ""
        lm = _LABEL_RE.match(text)
        name, count = (lm.group(1).strip(), int(lm.group(2).replace(",", ""))) if lm else (text, None)
        if name:
            out[param_to_field[m.group(1)]][norm_tag(name)] = {"id": inp.get("value", ""), "name": name, "count": count}
    return out


def _stat(dl, cls):
    node = dl.find("dd", class_=cls) if dl else None
    if node is None:
        return 0
    m = re.search(r"\d+", node.get_text(strip=True).replace(",", ""))
    return int(m.group()) if m else 0


def parse_blurbs(html, sep):
    """Every work card on a results page -> a dict shaped like a sample-CSV row."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for li in soup.select("li.work.blurb"):
        m = re.search(r"work_(\d+)", li.get("id", ""))
        link = li.select_one("h4.heading a[href^='/works/']")
        if not m and link:
            m = re.search(r"/works/(\d+)", link.get("href", ""))
        if not m:
            continue
        work_id = m.group(1)

        rating, categories = "", []
        req = li.find("ul", class_="required-tags")
        for span in (req.find_all("span") if req else []):
            title = span.get("title", "").strip()
            if not title:
                continue
            if title in RATING_VALUES:
                rating = title
                continue
            parts = [p.strip() for p in title.split(",")]
            if parts and all(p in CATEGORY_VALUES for p in parts):
                categories += [p for p in parts if p not in categories]

        def tags(css):
            return [a.get_text(" ", strip=True) for a in li.select(css) if a.get_text(strip=True)]

        stats = li.find("dl", class_="stats")
        lang = stats.find("dd", class_="language") if stats else None
        updated = li.find("p", class_="datetime")
        chap_node = stats.find("dd", class_="chapters") if stats else None
        cm = re.search(r"(\d+)\s*/", chap_node.get_text(" ", strip=True)) if chap_node else None
        hits, kudos = _stat(stats, "hits"), _stat(stats, "kudos")
        bookmarks, comments, words = _stat(stats, "bookmarks"), _stat(stats, "comments"), _stat(stats, "words")
        fandoms = tags("h5.fandoms a.tag")

        out.append({
            "work_id": work_id,
            "url": f"https://archiveofourown.org/works/{work_id}",
            "title": link.get_text(" ", strip=True) if link else "",
            "fandom": fandoms[0] if fandoms else "",
            "_fandoms": fandoms,   # every fandom tag on the card (not written to the CSV)
            "rating": rating,
            "warnings": sep.join(tags("ul.tags li.warnings a.tag")),
            "category": sep.join(categories),
            "relationships": sep.join(tags("ul.tags li.relationships a.tag")),
            "characters": sep.join(tags("ul.tags li.characters a.tag")),
            "freeform_tags": sep.join(tags("ul.tags li.freeforms a.tag")),
            "date_published": "",
            "date_updated": updated.get_text(" ", strip=True) if updated else "",
            "words": words, "chapters": int(cm.group(1)) if cm else 1,
            "language": lang.get_text(" ", strip=True) if lang else "",
            "hits": hits, "kudos": kudos, "bookmarks": bookmarks, "comments": comments,
            "kudos_to_hits": round(kudos / hits * 100, 4) if hits else 0,
            "bookmarks_to_hits": round(bookmarks / hits * 100, 4) if hits else 0,
            "comments_to_hits": round(comments / hits * 100, 4) if hits else 0,
            "comments_to_kudos": round(comments / kudos * 100, 4) if kudos else 0,
            "kudos_per_10k_words": round(kudos / words * 10000, 4) if words else 0,
            "bookmarks_per_10k_words": round(bookmarks / words * 10000, 4) if words else 0,
            "comments_per_10k_words": round(comments / words * 10000, 4) if words else 0,
            "tags_enriched": "",  # left blank so ao3_enrich_sample.py fills the rest from the work page
        })
    return out


# ══════════════════════════════════════════════════════════════
# STRATA, DATE WINDOWS, AND THE RANDOM DRAW
# ══════════════════════════════════════════════════════════════

class Spec:
    """A stratum: one fandom, optionally narrowed to one tag (rating / warning / category / character / relationship)."""

    def __init__(self, fandom, field=None, tag=None):
        self.fandom, self.field, self.tag = fandom, field, tag

    def key(self):
        return (f"{self.fandom}||{self.field or ''}||{self.tag or ''}||{FANDOM_AO3_TAG.get(self.fandom, '')}"
                f"||{CFG.crossover if CFG else ''}||{FANDOM_STYLE.get(self.fandom, 'tag_id')}")

    def label(self):
        return f"{self.fandom}" + (f" + {self.field}={self.tag}" if self.field else "")


class Stratum:
    def __init__(self, spec, leaves):
        self.spec = spec
        self.leaves = leaves  # [{"from": iso|None, "to": iso|None, "count": int}]

    @property
    def total(self):
        return sum(l["count"] for l in self.leaves)

    def locate(self, r):
        """Global index r -> (leaf, offset inside that leaf)."""
        acc = 0
        for leaf in self.leaves:
            if r < acc + leaf["count"]:
                return leaf, r - acc
            acc += leaf["count"]
        raise IndexError(r)


class StratumStore:
    def __init__(self, client, cache_path, ttl_days, date_mode):
        self.client = client
        self.path = cache_path
        self.ttl = ttl_days
        self.date_mode = date_mode  # "inclusive" | "exclusive" | "auto"
        self.mem = {}
        self.cache = {"strata": {}, "date_mode": None}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                self.cache.setdefault("strata", {})
            except (OSError, json.JSONDecodeError):
                pass

    def save(self):
        if self.path:
            try:
                Path(self.path).write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
            except OSError as err:
                print(f"  (could not write {self.path}: {err})")

    # -- URLs / counts ------------------------------------------------
    def to_param(self, d):
        """date_to as AO3 should receive it for an INCLUSIVE window ending on `d`."""
        return d + timedelta(days=1) if self.date_mode == "exclusive" else d

    # -- tag IDs (read from the fandom's own filter sidebar) -----------
    def sidebar_total(self, fandom):
        return (self.cache.get("sidebar", {}).get(fandom) or {}).get("total")

    def sidebar(self, fandom, force=False):
        """{field: {norm name: {"id","name","count"}}} from the UNFILTERED fandom page. Cached."""
        cached = self.cache.setdefault("sidebar", {}).get(fandom)
        key = FANDOM_AO3_TAG[fandom]
        if not force and cached and cached.get("tag") == key and self._age_days(cached.get("fetched_at")) <= self.ttl * 10:
            return cached["data"]
        url = BASE_URL + urllib.parse.urlencode([("commit", "Sort and Filter"), ("page", "1"), ("tag_id", key)])
        html = self.client.get_required(url)
        data = parse_sidebar(html)
        if not any(data.values()):
            raise TagIdError(f"Could not read the filter sidebar for '{key}'. Is the fandom tag name right? "
                             "(It is taken from the `fandom` column of your sample.)")
        self.cache["sidebar"][fandom] = {"tag": key, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "total": parse_work_count(html), "data": data}
        self.save()
        return data

    def tag_id(self, spec):
        if spec.field is None:
            return None
        hit = self.sidebar(spec.fandom).get(spec.field, {}).get(norm_tag(spec.tag))
        if hit:
            return hit["id"]  # authoritative: AO3's own checkbox value for this tag on this fandom page
        if spec.field in STATIC_IDS and spec.tag in STATIC_IDS[spec.field]:
            return STATIC_IDS[spec.field][spec.tag]  # fallback only
        raise TagIdError(f"'{spec.tag}' is not in the {spec.field} filter sidebar of '{FANDOM_AO3_TAG[spec.fandom]}', "
                         "so it has no tag ID to filter by.")

    def url(self, spec, page=1, d_from=None, d_to=None, style=None):
        style = style or FANDOM_STYLE.get(spec.fandom, "tag_id")
        tag = FANDOM_AO3_TAG[spec.fandom]
        if style == "scan":
            # AO3 does not apply filters for this fandom: list the WHOLE fandom; the card's own tags decide acceptance.
            return BASE_URL + urllib.parse.urlencode([("commit", "Sort and Filter"), ("page", str(page)),
                                                      ("work_search[sort_column]", "revised_at"),
                                                      ("work_search[sort_direction]", "desc"), ("tag_id", tag)])
        params = [("commit", "Sort and Filter"), ("page", str(page)),
                  ("work_search[sort_column]", "revised_at"), ("work_search[sort_direction]", "desc")]
        if style == "search":
            base = "https://archiveofourown.org/works/search?"
            params = [("utf8", "\u2713")] + params + [("work_search[fandom_names]", tag)]
            if spec.field == "rating":
                # On AO3's search form rating is a single <select>: work_search[rating_ids]=12 (no [])
                params.append(("work_search[rating_ids]", str(STATIC_IDS["rating"][spec.tag])))
            elif spec.field in ("warnings", "category"):
                names = {"warnings": "archive_warning_ids", "category": "category_ids"}
                params.append((f"work_search[{names[spec.field]}][]", str(STATIC_IDS[spec.field][spec.tag])))
            elif spec.field == "characters":
                params.append(("work_search[character_names]", spec.tag))
            elif spec.field == "relationships":
                params.append(("work_search[relationship_names]", spec.tag))
        else:
            if style == "tag_path":
                base = f"https://archiveofourown.org/tags/{encode_tag_path(tag)}/works?"
            else:
                base = BASE_URL
                params.append(("tag_id", tag))
            if spec.field:
                params.append((f"include_work_search[{FIELD_ID_PARAM[spec.field]}][]", str(self.tag_id(spec))))
        if CFG.crossover == "exclude":
            params.append(("work_search[crossover]", "F"))
        if d_from is not None:
            params.append(("work_search[date_from]", d_from.isoformat()))
        if d_to is not None:
            params.append(("work_search[date_to]", self.to_param(d_to).isoformat()))
        return base + urllib.parse.urlencode(params)

    # -- choose the URL form that really filters ---------------------
    def ensure_style(self, fandom):
        """Decide (once per run) which URL form filters for this fandom."""
        if not hasattr(self, "_styled"):
            self._styled = set()
        if fandom in self._styled:
            return FANDOM_STYLE[fandom]
        style = self._choose_style(fandom)
        self._styled.add(fandom)
        return style

    def _choose_style(self, fandom):
        """Probe one known rating on each URL form; keep the first whose count matches the sidebar."""
        key = FANDOM_AO3_TAG[fandom]
        cached = self.cache.setdefault("style", {}).get(fandom)
        if cached and cached.get("tag") == key and cached.get("crossover") == CFG.crossover and cached.get("style") in STYLES + ["scan"]:
            FANDOM_STYLE[fandom] = cached["style"]
            return cached["style"]
        sb = self.sidebar(fandom).get("rating", {})
        pool = [(v["count"], v["name"]) for v in sb.values() if v.get("count")]
        if not pool:
            FANDOM_STYLE[fandom] = "tag_id"
            return "tag_id"
        # a mid-sized rating: filtered count clearly differs from the fandom total
        pool.sort()
        expected, rating = pool[len(pool) // 2]
        probe = Spec(fandom, "rating", rating)
        seen = {}
        for style in STYLES:
            try:
                n = self.count(probe, style=style)
            except (SliceError, TagIdError, KeyError) as err:
                seen[style] = f"error: {err}"
                continue
            seen[style] = f"{n} (final URL: {getattr(self.client.fetcher, 'last_url', '') or 'n/a'})"
            if 0.4 * expected <= n <= 1.3 * expected + 50:
                FANDOM_STYLE[fandom] = style
                self.cache["style"][fandom] = {"tag": key, "crossover": CFG.crossover, "style": style,
                                               "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                self.save()
                note = "" if style == "tag_id" else f"  (the plain /works?tag_id= form did not apply filters: {seen.get('tag_id')})"
                print(f"  URL form for {fandom}: '{style}' filters correctly (probe {rating}: {n:,} vs sidebar ~{expected:,}){note}")
                return style
        # No URL form applies filters for this fandom. If the whole fandom fits inside AO3's 5,000-page window,
        # sample the WHOLE fandom uniformly and keep only works whose card carries the needed tag (rejection
        # sampling: still uniform within the tag, just slower).
        total = self.sidebar_total(fandom) or self.count(Spec(fandom), style="tag_id")
        detail = "; ".join(f"{k} -> {v}" for k, v in seen.items())
        if total and total <= SAFE_CAP:
            FANDOM_STYLE[fandom] = "scan"
            self.cache["style"][fandom] = {"tag": key, "crossover": CFG.crossover, "style": "scan",
                                           "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.save()
            print(f"  URL form for {fandom}: AO3 ignores filters here ({detail}).")
            print(f"    -> Falling back to SCAN mode: uniform draws from all {total:,} works, keeping only works whose card has "
                  f"the needed tag. Slower (expect ~1/share pages per accepted work) but still random.")
            return "scan"
        raise FilterError(f"AO3 ignores filters for {fandom} ({detail}) and the fandom ({total or '?'} works) is too big to scan. "
                          f"Run: python3 ao3_fix_sample.py --diagnose {fandom!r}  and send me the output.")

    def scan_share(self, spec):
        """Share of the fandom's works that carry this tag (from the sidebar)."""
        total = self.sidebar_total(spec.fandom) or 1
        hit = self.sidebar(spec.fandom).get(spec.field, {}).get(norm_tag(spec.tag)) if spec.field else None
        return (hit["count"] / total) if hit and hit.get("count") else 0.05

    def scan_pages(self, spec):
        """Page budget for ONE accepted work in SCAN mode: ~8x the expected number of pages (misses are rare)."""
        return int(min(800, max(30, 8 / max(self.scan_share(spec) * 0.8, 0.002))))

    def count(self, spec, d_from=None, d_to=None, style=None):
        html = self.client.get_required(self.url(spec, 1, d_from, d_to, style))
        c = parse_work_count(html)
        if c is None:
            raise SliceError(f"Could not read the 'Works found' line for {spec.label()} "
                             f"({d_from}..{d_to}). AO3 may have returned an unexpected page.")
        return c

    # -- date-boundary convention ------------------------------------
    def ensure_date_mode(self):
        if self.date_mode in ("inclusive", "exclusive"):
            return
        cached = self.cache.get("date_mode")
        if cached and cached.get("mode") in ("inclusive", "exclusive"):
            self.date_mode = cached["mode"]
            print(f"  Date-boundary convention (cached): date_to is {self.date_mode}.")
            return
        print("  Detecting whether AO3's date_to is inclusive (4 count queries)...")
        spec = Spec("Harry Potter")
        A = (date(2022, 1, 1), date(2022, 12, 31))
        B = (date(2023, 1, 1), date(2023, 12, 31))
        W = (date(2022, 1, 1), date(2023, 12, 31))
        self.date_mode = "inclusive"
        a, b, w = (self.count(spec, *A), self.count(spec, *B), self.count(spec, *W))
        tol = max(3, int(w * 0.0005))
        if abs(a + b - w) <= tol:
            mode = "inclusive"
        else:
            # If date_to is exclusive, [a,b) + [b,c) tiles [a,c) exactly. Re-test with that reading.
            self.date_mode = "exclusive"
            a2, b2, w2 = (self.count(spec, *A), self.count(spec, *B), self.count(spec, *W))
            if abs(a2 + b2 - w2) <= max(3, int(w2 * 0.0005)):
                mode = "exclusive"
            else:
                raise SliceError(
                    "Could not establish how AO3's date_to behaves (neither convention tiles cleanly: "
                    f"inclusive {a}+{b} vs {w}; exclusive {a2}+{b2} vs {w2}). "
                    "Refusing to sample, because uniformity cannot be guaranteed. Try --selftest.")
        self.date_mode = mode
        self.cache["date_mode"] = {"mode": mode, "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.save()
        print(f"  -> date_to is {mode}.")

    # -- building / loading strata -----------------------------------
    def get(self, spec):
        self.ensure_style(spec.fandom)  # must come first: the URL form is part of the cache key
        key = spec.key()
        if key in self.mem:
            return self.mem[key]
        entry = self.cache["strata"].get(key)
        if entry and self._age_days(entry.get("built_at")) <= self.ttl and entry.get("date_mode") in (None, self.date_mode, "auto"):
            st = Stratum(spec, entry["leaves"])
            try:
                self._sanity_check(spec, st.total)
                print(f"  Stratum [{spec.label()}]: {st.total:,} works in {len(st.leaves)} window(s) (cached).")
            except FilterError:
                print(f"  Cached stratum [{spec.label()}] looks wrong ({st.total:,}); rebuilding.")
                st = self.build(spec)
        else:
            st = self.build(spec)
        self.mem[key] = st
        return st

    @staticmethod
    def _age_days(iso):
        try:
            return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 86400
        except (TypeError, ValueError):
            return float("inf")

    def build(self, spec):
        self.ensure_style(spec.fandom)
        self.ensure_date_mode()
        print(f"  Building stratum [{spec.label()}] ...")
        total = self.count(spec)
        if total == 0:
            leaves = []
        elif total <= SAFE_CAP:
            leaves = [{"from": None, "to": None, "count": total}]
        else:
            leaves = []
            ranges = [(EARLIEST, date(2007, 12, 31))]
            ranges += [(date(y, 1, 1), date(y, 12, 31)) for y in range(2008, date.today().year + 1)]
            for a, b in ranges:
                self._collect(spec, a, b, leaves)
            got = sum(l["count"] for l in leaves)
            if abs(got - total) > max(5, int(total * 0.005)):
                raise SliceError(
                    f"Date windows for [{spec.label()}] add up to {got:,} but the unsliced total is {total:,}. "
                    "Windows are not tiling the stratum, so a uniform draw cannot be guaranteed. Stopping.")
            print(f"    {len(leaves)} windows; window counts add up to {got:,} vs total {total:,} (consistent).")
        self._sanity_check(spec, sum(l["count"] for l in leaves))
        st = Stratum(spec, leaves)
        self.cache["strata"][spec.key()] = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                            "date_mode": self.date_mode, "leaves": leaves}
        self.save()
        print(f"  Stratum [{spec.label()}]: {st.total:,} works.")
        return st

    def _sanity_check(self, spec, total):
        """A filtered stratum must be close to the sidebar's own count for that tag. If it is much bigger
        (typically = the whole fandom) AO3 ignored the filter; if much smaller the ID points at the wrong tag."""
        if not spec.field or FANDOM_STYLE.get(spec.fandom) == "scan":
            return
        hit = self.sidebar(spec.fandom).get(spec.field, {}).get(norm_tag(spec.tag))
        expected = hit["count"] if hit and hit.get("count") else None
        if not expected:
            return
        if total > 1.3 * expected + 50 or total < 0.4 * expected:
            raise FilterError(
                f"Stratum [{spec.label()}] has {total:,} works but the fandom sidebar says about {expected:,} for that tag. "
                f"AO3 {'ignored the filter' if total > expected else 'is filtering a different tag'} "
                f"(ID sent: {self.tag_id(spec)}). Not drawing from it. Run --selftest and send me its output.")

    def _collect(self, spec, a, b, leaves):
        c = self.count(spec, a, b)
        if c == 0:
            return
        if c <= SAFE_CAP:
            leaves.append({"from": a.isoformat(), "to": b.isoformat(), "count": c})
        elif a == b:
            raise SliceError(f"A single day ({a}) holds {c:,} works for [{spec.label()}], more than AO3 will show.")
        else:
            mid = a + (b - a) // 2
            self._collect(spec, a, mid, leaves)
            self._collect(spec, mid + timedelta(days=1), b, leaves)


class RandomSampler:
    def __init__(self, store, client, rng, sep):
        self.store, self.client, self.rng, self.sep = store, client, rng, sep
        self.rejections = Counter()
        self.draws = 0
        self._consecutive_tag_mismatch = 0

    def draw(self, spec, accept, max_pages=15):
        scan = FANDOM_STYLE.get(spec.fandom) == "scan"
        if scan:
            max_pages = self.store.scan_pages(spec)
        """One uniform draw from the stratum. `accept(candidate_row)` -> (ok, reason).
        Rejected draws are re-drawn; they are never adjusted toward a better-looking work."""
        st = self.store.get(spec)
        if st.total == 0:
            print(f"    Stratum [{spec.label()}] is empty on AO3; nothing to draw.")
            return None
        for _ in range(max_pages):
            r = self.rng.randrange(st.total)
            leaf, offset = st.locate(r)
            page, pos = offset // PAGE_SIZE + 1, offset % PAGE_SIZE
            d_from = date.fromisoformat(leaf["from"]) if leaf["from"] else None
            d_to = date.fromisoformat(leaf["to"]) if leaf["to"] else None
            html = self.client.get_required(self.store.url(spec, page, d_from, d_to))
            self.draws += 1

            live_count = parse_work_count(html)
            if live_count is not None and live_count != leaf["count"]:
                leaf["count"] = live_count  # the window drifted since it was counted
                self.store.save()
            blurbs = parse_blurbs(html, self.sep)
            window = f"{leaf['from'] or 'all'}..{leaf['to'] or 'all'}"
            where = f"r={r:,}/{st.total:,} -> window {window} offset {offset:,} -> page {page:,} pos {pos + 1}"
            if pos >= len(blurbs):
                print(f"    draw {where}: page shorter than expected (data moved); redrawing")
                self.rejections["page_shorter"] += 1
                continue
            cand = blurbs[pos]
            ok, reason = accept(cand)
            if not scan or ok:
                print(f"    draw {where}: work {cand['work_id']} -> {'ACCEPT' if ok else 'reject (' + reason + ')'}")
            elif self.draws % 10 == 0:
                print(f"    ...scanning (page draws so far this call: {_ + 1}, last reject: {reason})")
            if scan:
                pass
            elif reason == "tag_mismatch":
                self._consecutive_tag_mismatch += 1
                if self._consecutive_tag_mismatch >= 4:
                    raise FilterError(
                        f"AO3 keeps returning works that do not carry the filtered tag for [{spec.label()}]. "
                        "A filter name/ID is probably wrong. Run --selftest.")
            elif ok:
                self._consecutive_tag_mismatch = 0
            if ok:
                return cand
            self.rejections[reason] += 1
        print(f"    Gave up after {max_pages} pages without an acceptable work.")
        return None


# ══════════════════════════════════════════════════════════════
# BALANCE ENGINE (scores a swap without re-reading the whole CSV)
# ══════════════════════════════════════════════════════════════

class Balance:
    def __init__(self, checker, rows, sep):
        self.chk, self.rows, self.sep = checker, rows, sep
        self.prof, self.fandom_of = {}, {}
        self.idx = {}   # fandom -> [row index]
        self.agg = {}   # fandom -> {dim: [Counter, matched, raw]}
        for i, row in enumerate(rows):
            f = checker.detect_fandom(row)
            self.fandom_of[i] = f
            if f == "Unknown":
                continue
            self.prof[i] = self.profile(f, row)
            self.idx.setdefault(f, []).append(i)
        for f, ids in self.idx.items():
            self.agg[f] = {d[0]: [Counter(), 0, 0] for d in checker.DIMENSIONS}
            for i in ids:
                self._add(self.agg[f], self.prof[i], +1)

    def profile(self, fandom, row):
        info = self.chk.FANDOMS[fandom]
        prof = {}
        for dim, field, pop_key, _single in self.chk.DIMENSIONS:
            counts, matched, raw = self.chk.count_matched_tags([row], field, info[pop_key].keys(), self.sep)
            prof[dim] = (Counter({t: c for t, c in counts.items() if c}), matched, raw)
        return prof

    @staticmethod
    def _add(agg, prof, sign):
        for dim, (cnt, matched, raw) in prof.items():
            a = agg[dim]
            for t, c in cnt.items():
                a[0][t] += sign * c
            a[1] += sign * matched
            a[2] += sign * raw

    def stats(self, fandom, agg=None):
        agg = agg or self.agg[fandom]
        n = len(self.idx[fandom])
        info = self.chk.FANDOMS[fandom]
        out = {}
        for dim, field, pop_key, single in self.chk.DIMENSIONS:
            cnt, matched, raw = agg[dim]
            denom = n if single else (matched if matched > 0 else 1)
            out[dim] = self.chk.dimension_stats(cnt, denom, info[pop_key])
        return out

    def score(self, stats, dim):
        ss_dim = sum(t["diff"] ** 2 for t in stats[dim]["tags"])
        ss_all = sum(t["diff"] ** 2 for d in stats.values() for t in d["tags"])
        flags = sum(1 for d in stats.values() for t in d["tags"] if self.chk.is_flagged(t["flag"]))
        return ss_dim, ss_all, flags

    def swapped_agg(self, fandom, remove_i, cand_prof):
        agg = {d: [Counter(v[0]), v[1], v[2]] for d, v in self.agg[fandom].items()}
        self._add(agg, self.prof[remove_i], -1)
        self._add(agg, cand_prof, +1)
        return agg

    def apply(self, fandom, remove_i, new_row, cand_prof):
        self._add(self.agg[fandom], self.prof[remove_i], -1)
        self._add(self.agg[fandom], cand_prof, +1)
        self.prof[remove_i] = cand_prof
        self.rows[remove_i] = new_row


# ══════════════════════════════════════════════════════════════
# CSV I/O
# ══════════════════════════════════════════════════════════════

def save_rows(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def append_removed(record):
    exists = os.path.exists(REMOVED_FILE)
    with open(REMOVED_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record), quoting=csv.QUOTE_ALL)
        if not exists:
            w.writeheader()
        w.writerow(record)


# ══════════════════════════════════════════════════════════════
# PLANNING
# ══════════════════════════════════════════════════════════════

def pick_targets(chk, balance, skip, how_many):
    """The `how_many` worst (fandom, dimension) pairs that still contain a ! / !!! tag."""
    cands = []
    for f in balance.idx:
        stats = balance.stats(f)
        for dim, ds in stats.items():
            if (f, dim) in skip:
                continue
            worst = max((t["abs_diff"] for t in ds["tags"] if chk.is_flagged(t["flag"])), default=0)
            if worst > 0:
                cands.append((worst, f, dim))
    cands.sort(reverse=True)
    return [(f, dim, w) for w, f, dim in cands[:how_many]]


def over_under(ds):
    over = max(ds["tags"], key=lambda t: t["diff"])
    under = min(ds["tags"], key=lambda t: t["diff"])
    if over["diff"] <= 0 or under["diff"] >= 0:
        return None, None
    return over, under


def print_flag_table(chk, analysis, title):
    items = chk.flagged_items(analysis)
    print(f"\n{title}: {len(items)} tag(s) flagged ! or !!!")
    for it in items:
        print(f"    {it['flag'].strip():>3}  {it['fandom']:<13} {it['dim']:<13} {it['tag'][:46]:<46} "
              f"pop {it['pop_pct']:5.1f}%  sample {it['samp_pct']:5.1f}%  diff {it['diff']:+6.1f}")
    return items


# ══════════════════════════════════════════════════════════════
# ONE SWAP
# ══════════════════════════════════════════════════════════════

def attempt_swap(chk, balance, sampler, fieldnames, protected, blocked_ids, fandom, dim, field, sep):
    stats = balance.stats(fandom)
    ds = stats[dim]
    over, under = over_under(ds)
    if over is None:
        return None
    base_dim, base_all, base_flags = balance.score(stats, dim)

    pool = [i for i in balance.idx[fandom]
            if i not in protected and balance.prof[i][dim][0].get(over["tag"], 0) > 0]
    if not pool:
        print(f"    No removable works carry '{over['tag']}'.")
        return None

    print(f"  Swap: too many '{over['tag']}' ({over['diff']:+.1f} pp), too few '{under['tag']}' "
          f"({under['diff']:+.1f} pp) in {fandom} {dim}. {len(pool)} removable work(s).")

    spec = Spec(fandom, field, under["tag"])
    if FANDOM_STYLE.get(fandom) == "scan":
        share = sampler.store.scan_share(spec)
        expect = 1 / max(share * 0.8, 0.002)
        print(f"    SCAN mode: no AO3 filter; drawing from the whole fandom and keeping works tagged '{under['tag']}' "
              f"({share:.1%} of the fandom). Expect ~{expect:.0f} pages (~{expect * 12.5 / 60:.0f} min) per accepted work.")
    else:
        try:
            print(f"    Filter: include_work_search[{FIELD_ID_PARAM[field]}][]={sampler.store.tag_id(spec)} ({under['tag']})")
        except TagIdError as err:
            print(f"    {err}")
            return None

    in_sample = {r.get("work_id", "") for r in balance.rows}
    seen = set()

    def accept(c):
        if c["work_id"] in in_sample or c["work_id"] in blocked_ids or c["work_id"] in seen:
            return False, "duplicate"
        seen.add(c["work_id"])
        if chk.detect_fandom(c) != fandom:
            return False, "not_first_fandom"
        if not CFG.allow_related_fandoms:
            # Strict: the card must list the target fandom tag and NOTHING else. This keeps out e.g. 'Avatar: Legend of
            # Korra' (or works tagged both), which AO3 can treat as 'related' rather than a crossover.
            target = norm_tag(FANDOM_AO3_TAG[fandom])
            tags = [norm_tag(x) for x in c.get("_fandoms", [])]
            if not tags or any(x != target for x in tags):
                return False, "other_fandom_listed"
        if not eligible(c):
            return False, "ineligible"
        prof = balance.profile(fandom, c)
        if prof[dim][0].get(under["tag"], 0) == 0:
            return False, "tag_mismatch"
        return True, ""

    for _ in range(CFG.draws_per_swap):
        cand = sampler.draw(spec, accept)
        if cand is None:
            return None
        cprof = balance.profile(fandom, cand)
        ok_idx = []
        for i in pool:
            agg2 = balance.swapped_agg(fandom, i, cprof)
            s2 = balance.stats(fandom, agg2)
            d2, a2, f2 = balance.score(s2, dim)
            if d2 < base_dim - 1e-9 and a2 < base_all - 1e-9 and f2 <= base_flags:
                ok_idx.append(i)
        if not ok_idx:
            print(f"    Work {cand['work_id']} would not improve the balance for any removable work; drawing again.")
            continue
        i = sampler.rng.choice(ok_idx)  # random among the removals that help
        old = balance.rows[i]
        new_row = {k: "" for k in fieldnames}
        new_row.update({k: v for k, v in cand.items() if k in fieldnames or k in ENRICH_FIELDS_ORDER})
        new_row["sample_id"] = old.get("sample_id", "")
        balance.apply(fandom, i, new_row, cprof)
        in_sample.add(cand["work_id"])
        return {"index": i, "old": old, "new": new_row, "over": over["tag"], "under": under["tag"],
                "dim": dim, "fandom": fandom}
    print("    No draw produced an improving swap.")
    return None


def eligible(c):
    if CFG.min_hits and c.get("hits", 0) < CFG.min_hits:
        return False
    if CFG.min_words and c.get("words", 0) < CFG.min_words:
        return False
    if CFG.languages and c.get("language", "").strip().lower() not in CFG.languages:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# SELF-TEST (live AO3, ~15 requests)
# ══════════════════════════════════════════════════════════════

def selftest(chk, store, client):
    print("\n=== SELF-TEST: checking every assumption this script makes about AO3 ===")
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def near(live, pop, tol=0.10):
        return pop > 0 and abs(live - pop) / pop <= tol

    for f, info in chk.FANDOMS.items():
        n = store.count(Spec(f))
        if CFG.crossover == "exclude":
            ok = 0.6 * info["total"] <= n <= 1.1 * info["total"]
            note = f" (crossovers excluded: {1 - n / info['total']:.0%} of the sidebar total are removed)"
        else:
            ok, note = near(n, info["total"]), ""
        check(f"{f}: tag_id '{FANDOM_AO3_TAG[f]}'", ok, f"live {n:,} vs checker population {info['total']:,}{note}")

    # Every tag the checker tracks must exist in that fandom's sidebar (this is how IDs like 2774622 are found)
    for f, info in chk.FANDOMS.items():
        try:
            sb = store.sidebar(f)
        except TagIdError as err:
            check(f"{f}: filter sidebar readable", False, str(err))
            continue
        missing = [f"{x[1]}:{t}" for x in chk.DIMENSIONS for t in info[x[2]] if norm_tag(t) not in sb.get(x[1], {})
                   and not (x[1] in STATIC_IDS and t in STATIC_IDS[x[1]])]
        check(f"{f}: sidebar has an ID for every tag the checker tracks", not missing,
              "all found" if not missing else "missing -> " + "; ".join(missing[:6]))
        off = [(t, sb[x[1]][norm_tag(t)]["count"], c) for x in chk.DIMENSIONS if x[1] in ("characters", "relationships")
               for t, c in info[x[2]].items() if norm_tag(t) in sb[x[1]] and sb[x[1]][norm_tag(t)]["count"]
               and not near(sb[x[1]][norm_tag(t)]["count"], c, 0.15)]
        check(f"{f}: sidebar counts agree with checker population", not off,
              "within 15%" if not off else "; ".join(f"{t}: live {a:,} vs {b:,}" for t, a, b in off[:4]))

    for f in chk.FANDOMS:
        try:
            store.ensure_style(f)
        except FilterError as err:
            check(f"{f}: a URL form that applies filters exists", False, str(err))

    # Hard-coded IDs vs what AO3's sidebar actually uses (a mismatch makes AO3 silently ignore the filter)
    for f, info in chk.FANDOMS.items():
        try:
            sb = store.sidebar(f)
        except TagIdError:
            continue
        bad = []
        for field, ids in STATIC_IDS.items():
            for tag, static in ids.items():
                hit = sb.get(field, {}).get(norm_tag(tag))
                if hit and str(hit["id"]) != str(static):
                    bad.append(f"{tag}: sidebar says {hit['id']}, script had {static}")
        check(f"{f}: built-in rating/warning/category IDs match the sidebar", not bad,
              "all match" if not bad else "; ".join(bad) + "  (the script now uses the sidebar values)")

    # Probe EVERY rating (and every warning / category for Arcane): the filtered count must be a plausible
    # slice of the sidebar's own count, not the whole fandom.
    def probe(f, field, tag):
        sb = store.sidebar(f).get(field, {}).get(norm_tag(tag))
        expected = sb["count"] if sb and sb.get("count") else chk.FANDOMS[f][{"rating": "ratings", "warnings": "warnings", "category": "categories"}[field]].get(tag)
        try:
            tid = store.tag_id(Spec(f, field, tag))
            n = store.count(Spec(f, field, tag))
        except TagIdError as err:
            check(f"{f} {field} = {tag}", False, str(err))
            return
        ok = bool(expected) and 0.4 * expected <= n <= 1.3 * expected + 50
        check(f"{f} filter {FIELD_ID_PARAM[field]}={tid} ({tag})", ok,
              f"live {n:,} vs sidebar ~{expected:,}" + ("" if ok else "  <-- filter ignored or wrong tag"))

    for f in chk.FANDOMS:
        for tag in RATING_IDS:
            probe(f, "rating", tag)
    if CFG.full:
        for tag in WARNING_IDS:
            probe("Arcane", "warnings", tag)
        for tag in CATEGORY_IDS:
            probe("Arcane", "category", tag)
    else:
        print("        (warning/category sweeps skipped: they already passed on your data. Add --full to repeat them.)")

    info = chk.FANDOMS["Arcane"]
    probes = [("rating", "Explicit", "ratings"), ("warnings", "Major Character Death", "warnings"),
              ("category", "F/F", "categories"), ("characters", "Jinx (League of Legends)", "characters"),
              ("relationships", "Caitlyn/Vi (League of Legends)", "relationships")]
    for field, tag, key in probes:
        try:
            tid = store.tag_id(Spec("Arcane", field, tag))
            n = store.count(Spec("Arcane", field, tag))
        except TagIdError as err:
            check(f"filter {field} = {tag}", False, str(err))
            continue
        pop = info[key][tag]
        plausible = (0.6 * pop <= n <= 1.1 * pop) if CFG.crossover == "exclude" else near(n, pop, 0.15)
        check(f"filter include_work_search[{FIELD_ID_PARAM[field]}][]={tid} ({tag})", plausible,
              f"live {n:,} vs checker population {info[key][tag]:,} (crossover={CFG.crossover})")

    try:
        store.date_mode = "auto"
        store.cache["date_mode"] = None
        store.ensure_date_mode()
        check("date windows tile exactly", True, f"date_to is {store.date_mode}")
    except SliceError as err:
        check("date windows tile exactly", False, str(err))

    html = client.get_required(store.url(Spec("Arcane"), 1))
    blurbs = parse_blurbs(html, CFG.sep)
    check("results page shows 20 works", len(blurbs) == PAGE_SIZE, f"parsed {len(blurbs)}")
    if blurbs:
        # Judge the whole page, not card #1: results are sorted by last-updated, so the first card is often a
        # brand-new work with 0 hits and no category chosen yet, which is legitimate.
        n = len(blurbs)
        fields = ["title", "fandom", "rating", "warnings", "category", "characters", "relationships", "words", "hits"]
        have = {k: sum(1 for b in blurbs if b.get(k)) for k in fields}
        print("        cards with each field filled: " + ", ".join(f"{k} {have[k]}/{n}" for k in fields))
        dead = [k for k in ("title", "fandom", "rating", "warnings", "category", "words", "hits") if have[k] == 0]
        sparse = [k for k in ("category", "hits", "words") if 0 < have[k] < 0.5 * n]
        check("work-card fields parse (whole page)", not dead,
              "every key field is present on at least one card" if not dead else f"EMPTY on every card: {dead}")
        if sparse:
            print(f"        note: {sparse} are empty on more than half the cards; unusual, worth a look.")
        if dead:
            from bs4 import BeautifulSoup
            li = BeautifulSoup(html, "html.parser").select_one("li.work.blurb")
            print("        --- raw HTML of the first card (send this if the parse is wrong) ---")
            print("        " + re.sub(r"\s+", " ", str(li))[:1800])
        good = next((b for b in blurbs if b["hits"] and b["category"]), blurbs[0])
        print(f"        sample parse: {good['title'][:50]!r} / {good['fandom']!r} / rating={good['rating']!r} / "
              f"category={good['category']!r} / hits={good['hits']:,} / {good['relationships'][:50]!r}")
    print(f"\n{'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED - do not run the fixer until these are resolved'}"
          f" ({client.requests} AO3 requests used)")
    return all(results)


# ══════════════════════════════════════════════════════════════
# DIAGNOSE (what did AO3 actually return for each URL form?)
# ══════════════════════════════════════════════════════════════

def diagnose(chk, store, client, fetcher, fandom_arg):
    from bs4 import BeautifulSoup
    fandom = next((f for f in chk.FANDOMS if f.lower() == fandom_arg.lower()), None)
    if not fandom:
        print(f"Unknown fandom '{fandom_arg}'. Choose from: {', '.join(chk.FANDOMS)}")
        return 1
    sb = store.sidebar(fandom)
    ratings = sorted(((v["count"] or 0, v["name"]) for v in sb["rating"].values()))
    expected, rating = ratings[len(ratings) // 2]
    spec = Spec(fandom, "rating", rating)
    print(f"\n=== DIAGNOSE {fandom}: probing rating '{rating}' (sidebar says ~{expected:,}; whole fandom ~{sum(c for c, _ in ratings):,}) ===")
    for style in STYLES:
        url = store.url(spec, 1, style=style)
        html = client.get_required(url)
        soup = BeautifulSoup(html, "html.parser")
        heads = [re.sub(r"\s+", " ", h.get_text(" ", strip=True))[:110] for h in soup.select("h2.heading, h3.heading")][:4]
        ticked = [i.get("name", "") + "=" + i.get("value", "") for i in soup.select("input[type=checkbox][checked]")][:8]
        flash = [re.sub(r"\s+", " ", f.get_text(" ", strip=True))[:140] for f in soup.select("div.flash, .notice, .error")][:3]
        fname = f"diag_{fandom.replace(' ', '_')}_{style}.html"
        Path(fname).write_text(html, encoding="utf-8")
        print(f"\n  [{style}]")
        print(f"    requested : {url}")
        print(f"    final URL : {getattr(fetcher, 'last_url', '') or '(not available)'}")
        print(f"    headings  : {heads}")
        print(f"    count     : {parse_work_count(html)}")
        print(f"    ticked filter boxes in the returned sidebar: {ticked or 'none'}")
        print(f"    notices   : {flash or 'none'}")
        print(f"    saved     : {fname}")
    print("\nSend me this whole block. 'final URL' differing from 'requested' means AO3 redirected (and may have dropped the filters).")
    return 0


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flag and randomly repair an unrepresentative AO3 sample.")
    p.add_argument("--csv", default="ao3_testing_sample.csv")
    p.add_argument("--sep", default=" | ")
    p.add_argument("--summary", default="checker_summary.txt", help="Written (overwritten) by sample_checker.py.")
    p.add_argument("--checker", default=None, help="Path to sample_checker.py")
    p.add_argument("--scraper-path", default=None, help="Path to ao3statswithplaywright.py")
    p.add_argument("--cookie-file", default="ao3_cookie.txt")
    p.add_argument("--enrich-script", default=None, help="Path to ao3_enrich_sample.py")
    p.add_argument("--no-enrich", action="store_true", help="Skip the final ao3_enrich_sample.py pass.")
    p.add_argument("--worst", type=int, default=2, help="How many worst (fandom, dimension) pairs to fix per round.")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--swaps-per-dim", type=int, default=4, help="Max swaps per targeted dimension per round.")
    p.add_argument("--max-swaps", type=int, default=40, help="Hard cap on total swaps this run.")
    p.add_argument("--target-pp", type=float, default=4.0, help="Stop working on a dimension once its worst gap is <= this.")
    p.add_argument("--draws-per-swap", type=int, default=10, help="Random replacements to try before giving up on a swap.")
    p.add_argument("--seed", type=int, default=None, help="Reproducible draws. Default: OS entropy (truly random).")
    p.add_argument("--cache-days", type=float, default=3.0, help="How long window counts are reused.")
    p.add_argument("--date-semantics", choices=["auto", "inclusive", "exclusive"], default="auto")
    p.add_argument("--crossover", choices=["exclude", "any"], default="exclude",
                   help="exclude = work_search[crossover]=F (as in your reference URL). Use 'any' if your population "
                        "counts (sidebar totals) include crossovers and your sample does too.")
    p.add_argument("--fandom-tag", action="append", metavar="FANDOM=TAG",
                   help='Override a canonical fandom tag, e.g. --fandom-tag "Avatar=Avatar: The Last Airbender (Cartoon 2005)". Repeatable.')
    p.add_argument("--tags-from-csv", action="store_true",
                   help="Take each fandom tag from the sample's `fandom` column instead of the built-in canonical names.")
    p.add_argument("--no-refresh", action="store_true",
                   help="Do not refresh population numbers from the live AO3 sidebar (uses population_live.json / built-in).")
    p.add_argument("--allow-related-fandoms", action="store_true",
                   help="Allow replacements that list other fandom tags besides the target (e.g. Korra alongside ATLA). "
                        "Default: only works whose sole fandom tag is the target.")
    p.add_argument("--min-hits", type=int, default=0, help="Replacement eligibility (default: none).")
    p.add_argument("--min-words", type=int, default=0)
    p.add_argument("--languages", default="", help="Comma list, e.g. English. Default: any language.")
    p.add_argument("--dry-run", action="store_true", help="Show flags and the plan. No network, no changes.")
    p.add_argument("--diagnose", metavar="FANDOM", default=None,
                   help="Show what AO3 returns for each URL form for this fandom, e.g. --diagnose Avatar.")
    p.add_argument("--full", action="store_true", help="With --selftest: also sweep every warning and category (+12 requests).")
    p.add_argument("--selftest", action="store_true", help="Verify filter names, date logic and parsing against live AO3.")
    p.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    a = p.parse_args(argv)
    a.languages = {x.strip().lower() for x in a.languages.split(",") if x.strip()}
    return a


def main(argv=None, fetcher=None):
    global CFG
    CFG = parse_args(argv)
    original_stdout = sys.stdout
    sys.stdout = LogTee(LOG_FILE, original_stdout)
    try:
        return _run(fetcher)
    except (BlockedError, SliceError, FilterError, TagIdError) as err:
        print(f"\nSTOPPED: {err}")
        print("Whatever was already swapped is saved in the CSV. Nothing is left half-written.")
        return 2
    except KeyboardInterrupt:
        print("\nStopped by user. Completed swaps are already saved.")
        return 130
    finally:
        sys.stdout.close() if hasattr(sys.stdout, "close") else None
        sys.stdout = original_stdout


def base_tag(name):
    """'Avatar: The Last Airbender (Cartoon 2005)' -> 'avatar: the last airbender' (disambiguation suffix removed)."""
    return norm_tag(re.sub(r"\s*\([^)]*\)\s*$", "", name or ""))


def set_fandom_tags(chk, csv_path):
    """Built-in canonical tags are authoritative; --fandom-tag overrides one; --tags-from-csv restores the old
    behaviour (use the most common `fandom` text in the sample)."""
    for item in CFG.fandom_tag or []:
        name, _, tag = item.partition("=")
        key = next((f for f in chk.FANDOMS if f.lower() == name.strip().lower()), None)
        if not key or not tag.strip():
            sys.exit(f"ERROR: --fandom-tag expects FANDOM=TAG with FANDOM one of {list(chk.FANDOMS)}; got '{item}'.")
        FANDOM_AO3_TAG[key] = tag.strip()
    if CFG.tags_from_csv:
        counts = {}
        for row in chk.read_csv(csv_path):
            f = chk.detect_fandom(row)
            if f != "Unknown" and row.get("fandom"):
                counts.setdefault(f, Counter())[row["fandom"]] += 1
        for f, c in counts.items():
            FANDOM_AO3_TAG[f] = c.most_common(1)[0][0]
    print("  Fandom tags used for tag_id: " + "; ".join(f"{k} = '{v}'" for k, v in FANDOM_AO3_TAG.items()))


def refresh_population(chk, store):
    """Read each fandom's CURRENT sidebar counts from AO3, save them to population_live.json and apply them
    to the checker, so every flag is judged against today's numbers instead of stale hard-coded ones."""
    data = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "AO3 fandom filter sidebar (unfiltered)", "fandoms": {}}
    print("\nRefreshing population numbers from the live AO3 sidebar...")
    for f in chk.FANDOMS:
        try:
            sb = store.sidebar(f, force=True)
        except (TagIdError, BlockedError) as err:
            print(f"  {f}: could not refresh ({err}); keeping the previous numbers.")
            continue
        entry = {"total": store.sidebar_total(f), "tag": FANDOM_AO3_TAG[f]}
        for dim, field, key, _single in chk.DIMENSIONS:
            entry[key] = {}
            for tag in chk.FANDOMS[f][key]:
                hit = sb.get(field, {}).get(norm_tag(tag))
                if hit and hit.get("count"):
                    entry[key][tag] = hit["count"]
        data["fandoms"][f] = entry
        old_total = chk.FANDOMS[f]["total"]
        print(f"  {f}: total {old_total:,} -> {entry['total']:,}; "
              f"{sum(len(entry[k]) for k in entry if k not in ('total', 'tag'))} tag counts read (tag: {FANDOM_AO3_TAG[f]})")
    changed = chk.apply_population(data)
    try:
        Path("population_live.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"  Saved population_live.json ({changed} tag counts differ from the previous numbers). "
              f"sample_checker.py will use it automatically.")
    except OSError as err:
        print(f"  (could not save population_live.json: {err})")


def audit_sample_fandoms(chk, rows):
    """Which fandom tags do the sample's rows actually carry? Flags look-alikes such as Korra under 'Avatar'."""
    per, unknown = {}, Counter()
    for r in rows:
        f = chk.detect_fandom(r)
        if f == "Unknown":
            unknown[r.get("fandom", "") or "(blank)"] += 1
        else:
            per.setdefault(f, Counter())[r.get("fandom", "") or "(blank)"] += 1
    print("\nSAMPLE FANDOM AUDIT (the `fandom` column = first fandom tag on each work):")
    for f, c in per.items():
        target = norm_tag(FANDOM_AO3_TAG[f])
        exact = {k: v for k, v in c.items() if norm_tag(k) == target}
        short = {k: v for k, v in c.items() if k not in exact and base_tag(k) == base_tag(FANDOM_AO3_TAG[f])}
        other = {k: v for k, v in c.items() if k not in exact and k not in short}
        line = f"  {f}: {sum(c.values())} rows; exactly '{FANDOM_AO3_TAG[f]}': {sum(exact.values())}"
        if short:
            line += ("; same fandom, shorter tag text (kept, counted): "
                     + "; ".join(f"'{k}' x{v}" for k, v in short.items()))
        if other:
            line += f"; OTHER fandom tags counted as {f}: " + "; ".join(f"'{k}' x{v}" for k, v in Counter(other).most_common(6))
        print(line)
    if unknown:
        print("  Rows the checker does not assign to any fandom: " + "; ".join(f"'{k}' x{v}" for k, v in unknown.most_common(6)))


def _run(fetcher):
    chk = load_checker(CFG.checker)
    csv_path = os.path.abspath(CFG.csv)
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found.")

    set_fandom_tags(chk, csv_path)

    if CFG.diagnose:
        if fetcher is None:
            fetcher = LiveFetcher(CFG.scraper_path, CFG.cookie_file)
        client = Client(fetcher)
        store = StratumStore(client, STRATA_CACHE, CFG.cache_days, CFG.date_semantics)
        try:
            return diagnose(chk, store, client, fetcher, CFG.diagnose)
        finally:
            fetcher.close()

    if CFG.selftest:
        if fetcher is None:
            fetcher = LiveFetcher(CFG.scraper_path, CFG.cookie_file)
        client = Client(fetcher)
        store = StratumStore(client, STRATA_CACHE, CFG.cache_days, CFG.date_semantics)
        try:
            return 0 if selftest(chk, store, client) else 1
        finally:
            fetcher.close()

    if CFG.dry_run:
        return _run_main(chk, csv_path, None, None, None)
    if fetcher is None:
        fetcher = LiveFetcher(CFG.scraper_path, CFG.cookie_file)
    client = Client(fetcher)
    store = StratumStore(client, STRATA_CACHE, CFG.cache_days, CFG.date_semantics)
    try:
        return _run_main(chk, csv_path, fetcher, client, store)
    finally:
        fetcher.close()


def _run_main(chk, csv_path, fetcher, client, store):
    if store is not None and not CFG.no_refresh:
        refresh_population(chk, store)
    audit_sample_fandoms(chk, chk.read_csv(csv_path))
    print("\nStep 1: running sample_checker.py (writes " + CFG.summary + ", overwriting it)")
    analysis = chk.run_check(csv_path, CFG.sep, CFG.summary)
    items = print_flag_table(chk, analysis, "BEFORE")

    rows = chk.read_csv(csv_path)
    fieldnames = list(rows[0].keys()) if rows else list(ENRICH_FIELDS_ORDER)
    for extra in ENRICH_FIELDS_ORDER:
        if extra not in fieldnames:
            fieldnames.append(extra)

    if not items:
        print("\nNothing is flagged. The sample already matches the population within 5 pp everywhere. No changes made.")
        return 0

    balance = Balance(chk, rows, CFG.sep)
    small = {f: len(ids) for f, ids in balance.idx.items() if len(ids) < 60}
    if small:
        print("\nNOTE: " + ", ".join(f"{f} has only {n} works (one work = {100 / n:.1f} pp)" for f, n in small.items())
              + ". Gaps of 5 pp are within normal random noise at that size, and a swap cannot land closer than one work.")

    targets = pick_targets(chk, balance, set(), CFG.worst)
    print("\nPLAN: the two worst flagged (fandom, dimension) pairs right now:")
    for f, dim, w in targets:
        print(f"    {f} / {dim}  (worst gap {w:.1f} pp)")
    print(f"  Up to {CFG.swaps_per_dim} swap(s) per pair per round, {CFG.max_rounds} round(s), {CFG.max_swaps} swap(s) total.")

    if CFG.dry_run:
        for f, dim, w in targets:
            ds = balance.stats(f)[dim]
            over, under = over_under(ds)
            if over:
                print(f"    would swap OUT a random '{over['tag']}' work, IN a random live-AO3 work tagged '{under['tag']}' ({f})")
        print("\n--dry-run: no network used, no files changed (checker_summary.txt was refreshed).")
        return 0

    if not CFG.yes:
        if input("\nProceed with live AO3 draws and rewrite the CSV (a backup is made first)? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 0

    rng = random.Random(CFG.seed) if CFG.seed is not None else random.SystemRandom()
    print(f"\nRandom source: {'seeded (--seed ' + str(CFG.seed) + ')' if CFG.seed is not None else 'OS entropy (SystemRandom), not reproducible by design'}")

    try:
        backup = csv_path.replace(".csv", f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy2(csv_path, backup)
        print(f"Backup written: {backup}")

        sampler = RandomSampler(store, client, rng, CFG.sep)
        protected, blocked, skip, swaps_done = set(), set(), set(), []

        for rnd in range(1, CFG.max_rounds + 1):
            targets = pick_targets(chk, balance, skip, CFG.worst)
            if not targets or len(swaps_done) >= CFG.max_swaps:
                break
            print(f"\n--- Round {rnd}: fixing " + "; ".join(f"{f}/{d} ({w:.1f} pp)" for f, d, w in targets) + " ---")
            progress = False
            for f, dim, _w in targets:
                field = next(x[1] for x in chk.DIMENSIONS if x[0] == dim)
                for _ in range(CFG.swaps_per_dim):
                    if len(swaps_done) >= CFG.max_swaps:
                        break
                    ds = balance.stats(f)[dim]
                    if ds["max_diff"] <= CFG.target_pp:
                        print(f"  {f}/{dim} is now within {CFG.target_pp} pp.")
                        break
                    res = attempt_swap(chk, balance, sampler, fieldnames, protected, blocked, f, dim, field, CFG.sep)
                    if res is None:
                        skip.add((f, dim))
                        print(f"  Could not improve {f}/{dim} further; leaving it for this run.")
                        break
                    protected.add(res["index"])
                    blocked.add(res["old"].get("work_id", ""))
                    swaps_done.append(res)
                    progress = True
                    save_rows(csv_path, fieldnames, balance.rows)
                    append_removed({
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "sample_id": res["new"].get("sample_id", ""), "fandom": f, "dimension": dim,
                        "over_tag_removed": res["over"], "under_tag_added": res["under"],
                        "removed_work_id": res["old"].get("work_id", ""), "removed_title": res["old"].get("title", ""),
                        "removed_url": res["old"].get("url", ""),
                        "added_work_id": res["new"].get("work_id", ""), "added_title": res["new"].get("title", ""),
                        "added_url": res["new"].get("url", ""),
                    })
                    print(f"  SWAPPED #{res['new'].get('sample_id')}: removed work {res['old'].get('work_id')} "
                          f"-> added work {res['new'].get('work_id')} (CSV saved)")
            if not progress:
                print("No target could be improved this round; stopping.")
                break

        print(f"\nDone drawing: {len(swaps_done)} swap(s), {client.requests} AO3 request(s). "
              f"Rejected draws by reason: {dict(sampler.rejections) or 'none'}")
    finally:
        fetcher.close()

    if swaps_done and not CFG.no_enrich:
        enrich = Path(CFG.enrich_script) if CFG.enrich_script else HERE / "ao3_enrich_sample.py"
        if enrich.exists():
            print(f"\nStep 3: running {enrich.name} to fill in the full work-page data for the new works...")
            subprocess.run([sys.executable, str(enrich), "--csv", csv_path, "--cookie-file", CFG.cookie_file], check=False)
        else:
            print(f"\n(ao3_enrich_sample.py not found at {enrich}; new rows carry card data only, with tags_enriched blank.)")

    print("\nStep 4: final sample_checker.py run (overwrites " + CFG.summary + ")")
    final = chk.run_check(csv_path, CFG.sep, CFG.summary)
    left = print_flag_table(chk, final, "AFTER")
    print(f"\nFlagged tags: {len(items)} -> {len(left)}. Backup of the original: see above. "
          f"Removed works are listed in {REMOVED_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())