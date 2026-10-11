"""
Live AO3 supply counts and post-stratification weights for analyser_grapher.py.

What it does
------------
1. Loads your scraper (FanficCode/V4/ao3statswithplaywright.py) and reuses its
   browser, pacing (10-15 s between page loads), retry logic and URL builder to
   read the "N Works found" line from AO3 search pages.
2. Counts, per fandom: F/F, M/M, F/M, Gen, Multi, Other and "all works", using
   the SAME search filters your sample was scraped with (crossover, warnings,
   completion, timeframe) plus the scraper's hits >= 50 floor, so the counts
   describe the population your sample was drawn from.
3. Caches every count in ao3_supply_cache.json (re-used for 7 days) so repeated
   runs do not ping AO3 again.
4. Falls back to the cache (even if stale) and then to MANUAL_SUPPLY below if
   AO3 cannot be reached.
5. Turns the counts into category shares and reweights the "All works" group so
   that its category mix matches AO3's real mix.

Weighting method (raking)
-------------------------
A work belongs to every category it lists, and AO3's category counts work the
same way (a work tagged F/F and M/M is counted in both), so the true shares are
share_c = (works listing c) / (all works). Because categories overlap, simple
per-group weights do not work; instead every work's weight is adjusted
iteratively (raking / iterative proportional fitting) until the weighted share
of works listing each category matches AO3's. The weighted "All works" group is
a weighted resample (fixed seed) so every plot and summary can use it unchanged.
F/F, M/M and F/M are unaffected. Needs has_<category> columns from the grapher.
"""

import importlib.util
import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

SUPPLY_CACHE_FILE = "ao3_supply_cache.json"
SUPPLY_TTL_DAYS = 7

# The scraper drops works with fewer than 50 hits, so the sample cannot contain
# them. Counting the population with the same floor keeps the weights honest.
# Set to False to count every work regardless of hits.
APPLY_HITS_FLOOR = True
HITS_FLOOR = 50

PURE = ["F/F", "M/M", "F/M"]
LIVE_CATEGORIES = ["F/F", "M/M", "F/M", "Gen", "Multi", "Other"]
CATEGORY_IDS = {"F/F": "116", "M/M": "23", "F/M": "22", "Gen": "21", "Multi": "2246", "Other": "24"}
SCRAPER_FILENAME = "ao3statswithplaywright.py"

# Supply counts you gave me (no hits floor, date unknown). Keyed by the label the
# grapher uses for each fandom (first word of the fandom name). Used only when
# neither live counts nor a cache are available.
MANUAL_SUPPLY = {
    "Harry": {"M/M": 301366, "F/M": 268009, "Gen": 110397, "F/F": 62551, "Multi": 47894, "Other": 18475},
    "Arcane:": {"F/F": 27435, "M/M": 23941, "F/M": 13018, "Gen": 6733, "Multi": 3066, "Other": 2110},
    "Avatar:": {"F/M": 23843, "Gen": 16717, "M/M": 12702, "F/F": 7691, "Multi": 3516, "Other": 1411},
}


class SupplyError(Exception):
    pass


# ============================================================
# Reading AO3
# ============================================================

def load_scraper(explicit=None):
    """Import the scraper file by path (its main() only runs under __main__)."""
    here = Path(__file__).resolve().parent
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        Path("FanficCode/V4") / SCRAPER_FILENAME,
        here.parent / "FanficCode" / "V4" / SCRAPER_FILENAME,
        here / SCRAPER_FILENAME,
        Path("..") / "FanficCode" / "V4" / SCRAPER_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("ao3_scraper_module", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SupplyError(
        f"Could not find {SCRAPER_FILENAME}. Pass --scraper-path /path/to/{SCRAPER_FILENAME}."
    )


_COUNT_RE = re.compile(r"([\d,]+)\s+Works?\s+found", re.IGNORECASE)


def parse_work_count(html):
    """Total works from an AO3 search page: '1 - 20 of 301,366 Works found in ...'
    or '5 Works found'. Returns 0 for 'No results found', None if unreadable."""
    soup = BeautifulSoup(html, "html.parser")
    for heading in soup.select("h3.heading"):
        match = _COUNT_RE.search(heading.get_text(" ", strip=True))
        if match:
            return int(match.group(1).replace(",", ""))
    text = soup.get_text(" ", strip=True)
    match = _COUNT_RE.search(text)
    if match:
        return int(match.group(1).replace(",", ""))
    if re.search(r"No results found", text, re.IGNORECASE):
        return 0
    return None


def filters_from_df(uw):
    """The scraper filter settings the sample was collected with (most common
    value in the CSV), so the live counts use the same search."""
    def mode(col, default="0"):
        if col not in uw.columns:
            return default
        s = uw[col].dropna().astype(str)
        return s.mode().iloc[0] if len(s) else default

    return {
        "warning": mode("warning_filter"),
        "crossover": mode("crossover_filter"),
        "completion": mode("completion_filter"),
        "timeframe": mode("timeframe_filter"),
    }


def build_supply_url(scraper, fandom, category_label, filters):
    # Category "0" and warning "0" add no filter in the scraper's URL builder.
    base = scraper.build_search_url(
        fandom, "0", filters["warning"], filters["crossover"],
        filters["completion"], filters["timeframe"], page=1,
    )
    extra = []
    if category_label != "All":
        extra.append(("work_search[category_ids][]", CATEGORY_IDS[category_label]))
    if APPLY_HITS_FLOOR:
        extra.append(("work_search[hits]", f">{HITS_FLOOR - 1}"))
    return base + ("&" + urllib.parse.urlencode(extra) if extra else "")


# ============================================================
# Cache
# ============================================================

def _load_cache():
    path = Path(SUPPLY_CACHE_FILE)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache):
    try:
        Path(SUPPLY_CACHE_FILE).write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError as error:
        print(f"Warning: could not write {SUPPLY_CACHE_FILE}: {error}")


def _age_days(iso):
    try:
        then = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return float("inf")
    return (datetime.now(timezone.utc) - then).total_seconds() / 86400


def _cached(cache, url, max_age_days):
    entry = cache.get(url)
    if entry and _age_days(entry.get("fetched_at")) <= max_age_days:
        return entry
    return None


# ============================================================
# Collecting counts
# ============================================================

def _fetch_fandom(scraper, label, fandom, filters, cache, refresh):
    counts, dates, used_live = {}, [], False
    for cat in LIVE_CATEGORIES + ["All"]:
        url = build_supply_url(scraper, fandom, cat, filters)
        entry = None if refresh else _cached(cache, url, SUPPLY_TTL_DAYS)
        if entry is None:
            print(f"  Supply: {label} / {cat} ...")
            response = scraper.get_with_retry(url)
            if response is None or response.status_code != 200:
                code = getattr(response, "status_code", "no response")
                raise SupplyError(f"AO3 returned {code} for {label} / {cat}")
            count = parse_work_count(response.text)
            if count is None:
                raise SupplyError(f"could not find the 'Works found' line for {label} / {cat}")
            entry = {"count": count, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "fandom": fandom, "category": cat}
            cache[url] = entry
            _save_cache(cache)  # keep progress even if a later request fails
            used_live = True
        counts[cat] = entry["count"]
        dates.append(entry["fetched_at"])
    return counts, min(dates), used_live


def _stale_from_cache(scraper_free_builder, cache, fandom, filters):
    """Rebuild counts from the cache at any age, without touching the network."""
    counts, dates = {}, []
    for cat in LIVE_CATEGORIES + ["All"]:
        url = scraper_free_builder(fandom, cat, filters)
        entry = cache.get(url)
        if not entry:
            return None
        counts[cat] = entry["count"]
        dates.append(entry["fetched_at"])
    return counts, min(dates)


def collect_supply(uw, refresh=False, no_fetch=False, scraper_path=None, notes=None):
    """Return {"counts": {label: {category: n}}, "provenance": {label: str},
    "filters": {...}, "hits_floor": bool} or None if nothing is available."""
    notes = notes if notes is not None else []
    labels = sorted(uw["fandom_display"].dropna().unique())
    if not labels:
        return None
    fandom_map = {l: uw.loc[uw["fandom_display"] == l, "fandom"].mode().iloc[0] for l in labels}
    filters = filters_from_df(uw)

    if "chapter_filter" in uw.columns and (uw["chapter_filter"].astype(str) != "0").any():
        notes.append(
            "NOTE: your sample used a chapter filter, which the scraper applies after download, so the "
            "live supply counts cannot mirror it; weights are approximate."
        )

    cache = _load_cache()
    counts, provenance = {}, {}
    scraper = None

    if not no_fetch:
        try:
            scraper = load_scraper(scraper_path)
            print("\nFetching live AO3 supply counts with your scraper "
                  "(a browser window opens; this takes a few minutes the first time)...")
            for label, fandom in fandom_map.items():
                c, oldest, live = _fetch_fandom(scraper, label, fandom, filters, cache, refresh)
                counts[label] = c
                provenance[label] = (f"live AO3 counts, fetched {oldest[:10]}" if live
                                     else f"cached AO3 counts from {oldest[:10]}")
        except Exception as error:  # noqa: BLE001 - any failure falls back, loudly
            msg = f"WARNING: live supply fetch stopped ({error.__class__.__name__}: {error}). Falling back."
            print(msg)
            notes.append(msg)
        finally:
            browser = getattr(scraper, "BROWSER", None)
            if browser is not None:
                try:
                    browser.close()
                except Exception:  # noqa: BLE001
                    pass

    else:
        # Offline mode: the scraper is imported only to rebuild URLs for cache lookups.
        try:
            scraper = load_scraper(scraper_path)
        except Exception:  # noqa: BLE001
            scraper = None

    for label, fandom in fandom_map.items():
        if label in counts:
            continue
        builder = (lambda f, cat, flt: build_supply_url(scraper, f, cat, flt)) if scraper else None
        stale = _stale_from_cache(builder, cache, fandom, filters) if builder else None
        if stale:
            counts[label], oldest = stale
            provenance[label] = f"STALE cached AO3 counts from {oldest[:10]}"
        elif label in MANUAL_SUPPLY:
            counts[label] = dict(MANUAL_SUPPLY[label])
            provenance[label] = "manual counts supplied by you (undated, no hits floor)"
        else:
            notes.append(f"WARNING: no supply counts for '{label}'; its all-works line stays unweighted.")

    if not counts:
        return None
    return {"counts": counts, "provenance": provenance, "filters": filters, "hits_floor": APPLY_HITS_FLOOR}


# ============================================================
# Weights (raking to AO3's real category shares)
# ============================================================

def rake_weights(X, targets, max_iter=500, tol=1e-6, cap=15.0):
    """Find weights so the weighted share of works with each (yes/no) column of X
    matches `targets`. X is a works x categories boolean matrix and categories
    overlap (a work can list several), which is why this is iterative proportional
    fitting (raking) rather than a simple per-group weight.
    Returns (weights with mean 1, largest remaining miss, number of capped weights)."""
    n, k = X.shape
    w = np.ones(n)
    usable = [j for j in range(k) if 0 < targets[j] < 1 and X[:, j].any() and (~X[:, j]).any()]
    err = 0.0
    n_capped = 0
    for _ in range(max_iter):
        err = 0.0
        for j in usable:
            m = X[:, j]
            q = w[m].sum() / w.sum()
            if q <= 0 or q >= 1:
                continue
            w[m] *= targets[j] / q
            w[~m] *= (1 - targets[j]) / (1 - q)
        w = w / w.mean()
        capped = np.clip(w, 1 / cap, cap)
        n_capped = int((capped != w).sum())
        w = capped / capped.mean()
        err = max((abs(w[X[:, j]].sum() / w.sum() - targets[j]) for j in usable), default=0.0)
        if err < tol:
            break
    return w, err, n_capped


def apply_supply_weights(df, supply, all_label="All works", seed=2026):
    """Replace the 'All works' rows with a weighted resample whose category mix
    matches AO3's. Needs has_<category> columns. Returns (new_df, table, meta, notes)."""
    rng = np.random.default_rng(seed)
    is_all = df["category_clean"] == all_label
    pieces = [df[~is_all]]
    all_rows = df[is_all]
    table, meta, notes = [], {}, []

    pieces.append(all_rows[all_rows["fandom_display"].isna()])  # no fandom: untouched

    for fandom, g in all_rows.dropna(subset=["fandom_display"]).groupby("fandom_display"):
        counts = supply["counts"].get(fandom)
        if not counts:
            pieces.append(g)
            continue

        category_sum = sum(counts.get(c, 0) for c in LIVE_CATEGORIES)
        all_count = counts.get("All") or 0
        source = "AO3 all-works count"
        if all_count <= 0:
            all_count, source = category_sum, "ASSUMED = sum of category counts (no all-works count available)"
            notes.append(f"NOTE: {fandom}: no all-works count, so shares assume works list one category each.")
        elif category_sum / all_count > 1.02:
            notes.append(
                f"NOTE: {fandom}: category counts add up to {category_sum / all_count:.0%} of all works "
                f"because works list several categories (expected); the weighting accounts for this."
            )

        X = np.column_stack([g[f"has_{c}"].to_numpy(dtype=bool) for c in LIVE_CATEGORIES])
        targets = [min(counts.get(c, 0) / all_count, 0.999) for c in LIVE_CATEGORIES]
        w, err, n_capped = rake_weights(X, targets)
        if err > 0.01:
            notes.append(
                f"WARNING: {fandom}: weighting could not match every category share "
                f"(largest miss {err * 100:.1f} points); a category may be missing from your sample."
            )

        ess = float(w.sum() ** 2 / (w ** 2).sum())
        idx = rng.choice(len(g), size=len(g), replace=True, p=w / w.sum())
        pieces.append(g.iloc[idx])
        meta[fandom] = {
            "provenance": supply["provenance"].get(fandom, ""), "all_count": all_count,
            "all_count_source": source, "ess": ess, "n": len(g), "max_err": err, "n_capped": n_capped,
        }

        for j, c in enumerate(LIVE_CATEGORIES):
            mask = X[:, j]
            true_count = counts.get(c, 0)
            n_c = int(mask.sum())
            table.append({
                "fandom": fandom, "category": c, "true_count": true_count,
                "true_share": true_count / all_count,
                "sample_n": n_c, "sample_share": n_c / len(g),
                "weighted_share": float(w[mask].sum() / w.sum()),
                "rate": (n_c / true_count) if true_count else float("nan"),
                "avg_weight": float(w[mask].mean()) if n_c else float("nan"),
            })

    return pd.concat(pieces, ignore_index=True), table, meta, notes