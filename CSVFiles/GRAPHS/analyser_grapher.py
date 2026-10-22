"""
Exploratory analysis of ao3_results.csv: engagement patterns by relationship category.

Groups shown in every graph:
    F/F        every work that LISTS F/F (a work tagged F/F and M/M is in both groups)
    M/M        every work that lists M/M
    F/M        every work that lists F/M (AO3's name for "M/F")
    All works  every unique work in the CSV (Gen, Multi, Other and uncategorised
               works too), reweighted to AO3's real category mix when live or
               cached supply counts are available.
    Categories are read from BOTH the `category` and `warnings` columns,
    because the scraper can file AO3's multi-category label under warnings.

Extra outputs: 09 single- vs multi-category works, 10 joint regression
(needs statsmodels), 11 ship analysis. After the first run, fill in the
`canon` column of ship_canon_labels.csv and run again to add canon status.

Usage:
    python3 analyser_grapher.py [path-to-ao3_results.csv]

Outputs:
    figures_raw/*.png   the 7 figures
    summary.txt         appended on every run: a timestamped plain-language
                        summary of each figure, with numbers from your data
"""

import argparse
import sys
import warnings as pywarnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import traceback

pywarnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="AO3 engagement analysis by relationship category.")
parser.add_argument("csv", nargs="?", default="ao3_results.csv", help="path to ao3_results.csv")
parser.add_argument("--refresh-supply", action="store_true",
                    help="ignore cached supply counts and fetch them from AO3 again")
parser.add_argument("--no-fetch", action="store_true",
                    help="never contact AO3; use cached or manual supply counts")
parser.add_argument("--scraper-path", default=None,
                    help="path to ao3statswithplaywright.py (default: FanficCode/V4/)")
ARGS = parser.parse_args()
CSV_PATH = ARGS.csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from ao3_supply import collect_supply, apply_supply_weights
except ImportError as _err:  # ao3_supply.py not next to this file, or a package is missing
    collect_supply = apply_supply_weights = None
    # Python deletes the 'as' name when the except block ends, so keep the text.
    SUPPLY_IMPORT_ERROR = f"{_err.__class__.__name__}: {_err}"
OUTPUT_DIR = Path("figures_raw")
SUMMARY_FILE = "summary.txt"

sns.set_theme(style="whitegrid", context="talk")

ALL_LABEL = "All works"
WEIGHTED_ALL = {"on": False}  # set True once the all-works group is reweighted to true supply


def display_label(cat):
    if cat == ALL_LABEL and WEIGHTED_ALL["on"]:
        return "All works (weighted)"
    return cat

def safe(title, fn, *args):
    try:
        return fn(*args)
    except Exception:
        tb = traceback.format_exc()
        print(f"\n!!! {title} FAILED:\n{tb}")
        summary.add(f"{title} - FAILED", tb.splitlines())
        return None 

PURE_CATEGORIES = ["F/F", "M/M", "F/M"]
ALL_GROUPS = PURE_CATEGORIES + [ALL_LABEL]

CATEGORY_PALETTE = {
    "F/F": "#D95F02",
    "M/M": "#1B9E77",
    "F/M": "#7570B3",
    ALL_LABEL: "#666666",
}
# Mutated in place by set_groups() so only groups present in the data are drawn.
CATEGORY_ORDER = list(ALL_GROUPS)
HUE_OFFSETS = {}

# Pairs whose median gap / statistical comparison is reported.
GAP_PAIRS = [("M/M", "F/F"), ("M/M", "F/M"), ("F/F", "F/M")]

FLIER_PROPS = dict(marker="o", markersize=3, alpha=0.35, markeredgewidth=0)
TICK_PAD = 34  # leaves room for two staggered rows of n= labels


def set_groups(present):
    """Restrict the drawn groups to those that exist, and recompute the
    horizontal offset seaborn gives each group inside a dodged box plot."""
    CATEGORY_ORDER[:] = [g for g in ALL_GROUPS if g in present]
    n = max(1, len(CATEGORY_ORDER))
    width = 0.8 / n
    HUE_OFFSETS.clear()
    for i, g in enumerate(CATEGORY_ORDER):
        HUE_OFFSETS[g] = -0.4 + width * (i + 0.5)


set_groups(ALL_GROUPS)

ENGAGEMENT_METRICS = [
    ("kudos_to_hits", "Kudos / Hits (%)"),
    ("comments_to_hits", "Comments / Hits (%)"),
    ("bookmarks_to_hits", "Bookmarks / Hits (%)"),
    ("comments_to_kudos", "Comments / Kudos (%)"),
]
WORD_NORMALIZED_METRICS = [
    ("kudos_per_10k_words", "Kudos per 10k words"),
    ("bookmarks_per_10k_words", "Bookmarks per 10k words"),
    ("comments_per_10k_words", "Comments per 10k words"),
]

WORD_Y_LIMITS = {
    "kudos_per_10k_words": (0, 1500),
    "bookmarks_per_10k_words": (0, 300),
    "comments_per_10k_words": (0, 150),
}

MIN_WORDS_FOR_NORMALIZATION = 500
MIN_GROUP_N_FOR_TEST = 15
MIN_N_NOISY = 10  # groups smaller than this are flagged as noise

# Scraper filter columns that should match across the pure categories for the
# comparison to be fair (category_filter is expected to differ).
FILTER_COLUMNS_TO_MATCH = [
    "warning_filter", "crossover_filter", "chapter_filter",
    "completion_filter", "timeframe_filter",
]

PROXY_CAVEAT = "CRUDE PROXY -- not your actual coded variable. Hypothesis-generating only."


# ============================================================
# Summary collector (written to summary.txt at the end of a run)
# ============================================================

class Summary:
    def __init__(self):
        self.sections = []

    def add(self, title, lines):
        self.sections.append((title, lines))

    def write(self, path, header_lines):
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 72 + "\n")
            f.write(f"RUN: {stamp}\n")
            for line in header_lines:
                f.write(line + "\n")
            f.write("=" * 72 + "\n")
            for title, lines in self.sections:
                f.write(f"\n### {title}\n")
                for line in lines:
                    f.write(line + "\n")
        print(f"Summary appended to {path} ({stamp}).")


def unique_works(df):
    """One row per work (the 'All works' rows are exactly the unique set)."""
    return df[df["category_clean"] == ALL_LABEL]


def _fmt_cat(group, col, cat):
    v = group.loc[group["category_clean"] == cat, col].dropna()
    if len(v) == 0:
        return f"{display_label(cat)}: no data", None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    flag = "  [n<10: noisy]" if len(v) < MIN_N_NOISY else ""
    return f"{display_label(cat)}: median {med:.2f} (IQR {q1:.2f}-{q3:.2f}, n={len(v)}){flag}", med


def compare_lines(df, col, label, by):
    """Per group: median/IQR/n for every category, then the median gaps."""
    lines = [f"  {label}:"]
    if df.empty:
        return lines + ["    (no data)"]
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        lines.append(f"    [{' / '.join(map(str, key))}]")
        meds = {}
        for cat in CATEGORY_ORDER:
            txt, med = _fmt_cat(g, col, cat)
            meds[cat] = med
            lines.append("      " + txt)
        gaps = []
        for a, b in GAP_PAIRS:
            if meds.get(a) is not None and meds.get(b) is not None:
                gaps.append(f"{a} minus {b}: {meds[a] - meds[b]:+.2f}")
        if gaps:
            lines.append("      median gaps -> " + "; ".join(gaps))
    return lines


# ============================================================
# Helper & Formatting Utilities
# ============================================================

def text_work_subset(df):
    """Text works long enough for per-10k-word ratios to be stable.
    Used by BOTH Plot 3 and the statistical tests so they agree."""
    return df[(df["is_text_work"] == True) & (df["words"] >= MIN_WORDS_FOR_NORMALIZATION)].copy()  # noqa: E712


def annotate_n(ax, x, n, level=0):
    """n= label under a box. `level` staggers neighbouring labels vertically
    so four groups do not print on top of each other."""
    color = "red" if n < MIN_N_NOISY else "dimgray"
    ax.text(
        x, -0.015 - 0.045 * level, f"n={n}", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=8, color=color, clip_on=False,
    )


def annotate_group_counts(ax, k, count_fn):
    for level, (cat, offset) in enumerate(HUE_OFFSETS.items()):
        annotate_n(ax, k + offset, count_fn(cat), level % 2)


def category_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=CATEGORY_PALETTE[c], edgecolor="black", label=display_label(c)) for c in CATEGORY_ORDER]


def use_shared_legend(fig, axes):
    for ax in np.ravel(axes):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(
        handles=category_legend_handles(), title="Category",
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True,
    )


def set_robust_ylim(ax, values, note=True):
    """Clip the view to roughly the 1st-99th percentile. Reports points
    hidden on BOTH sides. `values` should be one row per work."""
    values = pd.Series(values).dropna()
    if len(values) < 10:
        return 0, 0
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return 0, 0
    pad = (hi - lo) * 0.12
    y_lo = max(0, lo - pad) if lo >= 0 else lo - pad
    y_hi = hi + pad
    ax.set_ylim(y_lo, y_hi)
    n_above = int((values > y_hi).sum())
    n_below = int((values < y_lo).sum())
    if note and (n_above or n_below):
        ax.text(
            0.99, 0.99, f"{n_above} above / {n_below} below view (all works)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray",
        )
    return n_above, n_below


# ============================================================
# Reading category / warning labels
# ============================================================
#
# AO3 shows ONE icon per required tag, and for a work with several categories
# its title is a comma-joined string ("F/F, M/M"). The scraper only recognises
# single labels, so such strings can end up in the `warnings` column, and
# labels can generally land in the wrong one of the two columns. So both
# columns are read, split on commas, and every label is sorted by what it is.

CATEGORY_NAMES = ["F/F", "M/M", "F/M", "Gen", "Multi", "Other"]
CATEGORY_TOKENS = {"f/f": "F/F", "m/m": "M/M", "f/m": "F/M", "gen": "Gen", "multi": "Multi", "other": "Other"}
NO_CATEGORY_TOKENS = {"", "no category", "none"}
WARNING_TOKENS = {
    "no archive warnings apply": "No Archive Warnings Apply",
    "creator chose not to use archive warnings": "Creator Chose Not To Use Archive Warnings",
    "graphic depictions of violence": "Graphic Depictions Of Violence",
    "major character death": "Major Character Death",
    "rape/non-con": "Rape/Non-Con",
    "underage": "Underage",
    "underage sex": "Underage Sex",
}


def parse_labels(category_text, warnings_text):
    """Sort every label in both columns into categories or warnings.
    Returns (categories, warnings, category_was_in_warnings_column,
    warning_was_in_category_column, unrecognised_labels)."""
    cats, warns, unknown = [], [], []
    moved_cat = moved_warn = False
    for source, text in (("category", category_text), ("warnings", warnings_text)):
        if not isinstance(text, str):
            continue
        for token in text.split(","):
            token = token.strip()
            key = token.lower()
            if key in NO_CATEGORY_TOKENS:
                continue
            if key in CATEGORY_TOKENS:
                name = CATEGORY_TOKENS[key]
                if name not in cats:
                    cats.append(name)
                moved_cat = moved_cat or source == "warnings"
            elif key in WARNING_TOKENS:
                name = WARNING_TOKENS[key]
                if name not in warns:
                    warns.append(name)
                moved_warn = moved_warn or source == "category"
            else:
                # Unknown label: keep it with the warnings (where the scraper put it) and report it.
                if token not in warns:
                    warns.append(token)
                unknown.append(token)
    return cats, warns, moved_cat, moved_warn, unknown


# ============================================================
# Load
# ============================================================

def load_data(path, summary_notes):
    df = pd.read_csv(
        path,
        dtype={
            "work_id": str, "category_filter": str, "warning_filter": str,
            "crossover_filter": str, "chapter_filter": str,
            "completion_filter": str, "timeframe_filter": str,
        },
    )

    df = df[df["role"] == "cohort"].copy()
    n_before = len(df)
    df = df.dropna(subset=["work_id"])
    df = df.drop_duplicates(subset=["work_id"], keep="first")
    summary_notes.append(f"Cohort rows: {n_before}; duplicates / rows without a work_id dropped: {n_before - len(df)}.")

    # --- categories and warnings, read from BOTH columns ---
    parsed = [parse_labels(c, w) for c, w in zip(df["category"], df["warnings"])]
    for name in CATEGORY_NAMES:
        df[f"has_{name}"] = [name in p[0] for p in parsed]
    df["n_categories"] = [len(p[0]) for p in parsed]
    df["categories"] = ["|".join(sorted(p[0])) for p in parsed]
    df["warnings_clean"] = [", ".join(p[1]) for p in parsed]

    n_moved_cat = sum(p[2] for p in parsed)
    n_moved_warn = sum(p[3] for p in parsed)
    n_multi = int((df["n_categories"] > 1).sum())
    n_none = int((df["n_categories"] == 0).sum())
    summary_notes.append(
        f"Category labels found in the `warnings` column: {n_moved_cat} works; warning labels found in the "
        f"`category` column: {n_moved_warn} works (both columns were read and re-sorted)."
    )
    summary_notes.append(
        f"Works listing several categories: {n_multi} (each counts in EVERY category it lists); "
        f"works with no category at all: {n_none}."
    )
    unknown = pd.Series([u for p in parsed for u in p[4]])
    if len(unknown):
        top = unknown.value_counts().head(6)
        msg = ("NOTE: labels that are neither a known category nor a known warning (kept with the warnings): "
               + "; ".join(f"'{k}' x{v}" for k, v in top.items()))
        print(msg)
        summary_notes.append(msg)

    summary_notes.append("Works listing each category (a work can appear in several rows):")
    summary_notes.append("  " + "; ".join(f"{n}: {int(df[f'has_{n}'].sum())}" for n in CATEGORY_NAMES))
    for a, b in GAP_PAIRS:
        both = int((df[f"has_{a}"] & df[f"has_{b}"]).sum())
        summary_notes.append(f"  tagged with both {a} and {b}: {both}")

    # Stack: one copy of each work per category it lists, plus every unique work as 'All works'.
    pieces = []
    for cat in PURE_CATEGORIES:
        part = df[df[f"has_{cat}"]].copy()
        part["category_clean"] = cat
        pieces.append(part)
    everything = df.copy()
    everything["category_clean"] = ALL_LABEL
    pieces.append(everything)
    df = pd.concat(pieces, ignore_index=True)

    present = set(df["category_clean"].unique())
    set_groups(present)
    missing = [c for c in PURE_CATEGORIES if c not in present]
    if missing:
        msg = f"NOTE: no works found for {missing}; those groups are left out of the plots."
        print(msg)
        summary_notes.append(msg)

    if "fandom_display" not in df.columns:
        df["fandom_display"] = df["fandom"]

    # is_text_work is rebuilt from the word count (the scraper's own rule: words > 0).
    # Rows scraped by an older script version can have this column blank/False even
    # though 'words' is fine, which silently removed whole fandoms from Plots 3 and 6.
    df["words"] = pd.to_numeric(df["words"], errors="coerce")
    derived = df["words"].fillna(0) > 0
    if "is_text_work" in df.columns:
        stored = df["is_text_work"].astype(str).str.strip().str.lower() == "true"
        n_mismatch = int((derived != stored).sum())
        if n_mismatch:
            msg = (f"NOTE: stored is_text_work disagreed with words>0 on {n_mismatch} row(s) "
                   f"(counted across all groups); using words>0.")
            print(msg)
            summary_notes.append(msg)
    df["is_text_work"] = derived

    # Same grouping as before (first word of the fandom name), so the data
    # being analysed is unchanged -- but collisions are reported.
    df["fandom_display"] = df["fandom_display"].fillna("").str.strip().str.split().str[0]

    uw = unique_works(df)
    n_no_fandom = int(uw["fandom_display"].isna().sum())
    if n_no_fandom:
        msg = (f"WARNING: {n_no_fandom} work(s) have no fandom label and are left out of "
               f"every per-fandom plot.")
        print(msg)
        summary_notes.append(msg)

    collisions = (
        uw.dropna(subset=["fandom_display"])
        .groupby("fandom_display")["fandom"].agg(lambda s: sorted(set(s.dropna())))
    )
    for label, full_names in collisions.items():
        if len(full_names) > 1:
            msg = (f"WARNING: label '{label}' merges {len(full_names)} different fandom names: "
                   f"{full_names}. They are pooled together in every plot.")
            print(msg)
            summary_notes.append(msg)

    df["date_updated_parsed"] = pd.to_datetime(
        df["date_updated"], format="mixed", dayfirst=True, errors="coerce"
    )
    return df


def check_filter_consistency(df, summary_notes):
    """Report (does not change the data) whether the pure categories came
    from the same scraper filter settings."""
    problems = []
    pure_present = [c for c in PURE_CATEGORIES if c in CATEGORY_ORDER]
    for col in FILTER_COLUMNS_TO_MATCH:
        if col not in df.columns:
            continue
        per_cat = {
            cat: sorted(set(df.loc[df["category_clean"] == cat, col].dropna().astype(str)))
            for cat in pure_present
        }
        if len({tuple(v) for v in per_cat.values()}) > 1:
            problems.append(f"{col}: " + ", ".join(f"{c} used {v}" for c, v in per_cat.items()))

    if problems:
        print("\nWARNING: categories were scraped under different filters:")
        summary_notes.append("WARNING: categories were scraped under DIFFERENT filters (confound):")
        for p in problems:
            print("  - " + p)
            summary_notes.append("  - " + p)
    else:
        msg = ("Filter check: all categories share the same warning/crossover/chapter/"
               "completion/timeframe filters.")
        print("\n" + msg)
        summary_notes.append(msg)

    if df["date_updated_parsed"].notna().any():
        for cat in CATEGORY_ORDER:
            d = df.loc[df["category_clean"] == cat, "date_updated_parsed"].dropna()
            if len(d):
                summary_notes.append(
                    f"date_updated range for {cat}: {d.min().date()} to {d.max().date()} (n={len(d)})."
                )


def summarize_sample(df, summary_notes):
    print("=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)
    counts = (
        df.groupby(["fandom_display", "category_clean"]).size().unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )
    print(counts)
    print()
    summary_notes.append("Works per fandom x group:")
    for line in counts.to_string().splitlines():
        summary_notes.append("  " + line)

    uw = unique_works(df)
    for fandom, g in uw.dropna(subset=["fandom_display"]).groupby("fandom_display"):
        n_text = int(g["is_text_work"].sum())
        n_long = int(((g["is_text_work"]) & (g["words"] >= MIN_WORDS_FOR_NORMALIZATION)).sum())
        if n_long == 0:
            msg = (f"WARNING: '{fandom}' has no works with >= {MIN_WORDS_FOR_NORMALIZATION} words "
                   f"({n_text} with words>0), so it is absent from Plots 3, 6 and the per-10k-word tests. "
                   f"The 'words' column was probably not captured for these rows.")
            print(msg)
            summary_notes.append(msg)


# ============================================================
# Plot 1: ECDFs
# ============================================================

def plot_ecdfs(df, output_dir, summary):
    fandoms = df["fandom_display"].dropna().unique()
    fig, axes = plt.subplots(len(fandoms), len(ENGAGEMENT_METRICS), figsize=(22, 6 * len(fandoms)), squeeze=False)
    column_max = {col: df[col].max() for col, _ in ENGAGEMENT_METRICS}

    for i, fandom in enumerate(fandoms):
        fdf = df[df["fandom_display"] == fandom]
        for j, (col, label) in enumerate(ENGAGEMENT_METRICS):
            ax = axes[i][j]
            for cat in CATEGORY_ORDER:
                color = CATEGORY_PALETTE[cat]
                values = fdf.loc[fdf["category_clean"] == cat, col].dropna().sort_values()
                if len(values) == 0:
                    continue
                y = np.arange(1, len(values) + 1) / len(values)
                is_all = cat == ALL_LABEL
                ax.plot(
                    values, y, label=f"{display_label(cat)} (n={len(values)})", color=color,
                    linewidth=2.5 if is_all else 2, linestyle="--" if is_all else "-",
                    alpha=0.9 if is_all else 1.0,
                )
            ax.set_title(f"{fandom}\n{label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Cumulative proportion")
            ax.legend(title="Category (n)", fontsize=9, title_fontsize=9, loc="lower right")
            # Scale first, then limits (setting the scale afterwards can reset them).
            if (fdf[col] > 0).any():
                ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(0, column_max[col] * 1.02)

    fig.suptitle("Engagement distributions by category (ECDF; dashed = all works)", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_dir / "01_engagement_ecdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '01_engagement_ecdfs.png'}")

    lines = [
        "WHAT IT SHOWS: for each fandom (row) and ratio (column), the share of works at or below",
        "each value, for F/F, M/M, F/M and all works (dashed grey). A curve further RIGHT = generally",
        "higher engagement; a curve that rises quickly on the left = many low-engagement works. The",
        "x-axis is symlog (linear below 1, log above), so differences near zero are not exaggerated.",
        "NUMBERS (median/IQR/n per group):",
    ]
    for col, label in ENGAGEMENT_METRICS:
        lines += compare_lines(df, col, label, by=["fandom_display"])
    lines.append("CAUTION: ECDFs show the whole distribution; a median gap alone can hide a tail difference.")
    lines.append("'All works' contains the other three groups plus Gen/Multi/Other, so it sits between them by")
    lines.append("construction and is a reference, not a fourth independent category.")
    summary.add("01_engagement_ecdfs.png - Engagement distributions (ECDF)", lines)


# ============================================================
# Plot 2: Box plots by fandom and metric
# ============================================================

def plot_distributions(df, output_dir, summary):
    fig, axes = plt.subplots(1, len(ENGAGEMENT_METRICS), figsize=(26, 8))
    hidden = []
    uw = unique_works(df)
    for ax, (col, label) in zip(axes, ENGAGEMENT_METRICS):
        sns.boxplot(
            data=df, x="fandom_display", y=col, hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        n_above, n_below = set_robust_ylim(ax, uw[col])
        hidden.append(f"{label}: {n_above} above / {n_below} below the visible range (counted over all works)")
        ax.tick_params(axis="x", pad=TICK_PAD)

        for k, fandom in enumerate(df["fandom_display"].dropna().unique()):
            annotate_group_counts(
                ax, k,
                lambda cat, f=fandom: ((df["fandom_display"] == f) & (df["category_clean"] == cat)).sum(),
            )

    use_shared_legend(fig, axes)
    fig.suptitle("Engagement ratios by category and fandom (box = median/IQR)", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "02_engagement_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '02_engagement_boxplots.png'}")

    lines = [
        "WHAT IT SHOWS: the typical (median) and middle-50% (IQR) kudos, comment and bookmark",
        "rate per hit, plus comments per kudos, for F/F, M/M, F/M and all works within each fandom.",
        "Dots are outliers. The y-axis is clipped to about the 1st-99th percentile of all works.",
        "NUMBERS (median/IQR/n per group):",
    ]
    for col, label in ENGAGEMENT_METRICS:
        lines += compare_lines(df, col, label, by=["fandom_display"])
    lines.append("POINTS OUTSIDE THE VIEW (still in the data, just not drawn):")
    lines += ["  " + h for h in hidden]
    lines.append("CAUTION: red n labels (n<10) are noise. Bookmarks/hits is the stronger enjoyment")
    lines.append("signal; comments reflect community norms as much as enjoyment.")
    summary.add("02_engagement_boxplots.png - Engagement ratios by category and fandom", lines)


# ============================================================
# Plot 3: Word-normalized metrics (text works only)
# ============================================================

def plot_word_normalized(df, output_dir, summary):
    text_df = text_work_subset(df)
    uw = unique_works(df)
    n_excluded = int(((uw["is_text_work"] == True) & (uw["words"] < MIN_WORDS_FOR_NORMALIZATION)).sum())  # noqa: E712
    uw_text = unique_works(text_df)

    fig, axes = plt.subplots(1, len(WORD_NORMALIZED_METRICS), figsize=(24, 8))
    hidden = []
    for ax, (col, label) in zip(axes, WORD_NORMALIZED_METRICS):
        sns.boxplot(
            data=text_df, x="fandom_display", y=col, hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")

        if col in WORD_Y_LIMITS:
            ax.set_ylim(WORD_Y_LIMITS[col])
            n_above = int((uw_text[col] > WORD_Y_LIMITS[col][1]).sum())
            n_below = int((uw_text[col] < WORD_Y_LIMITS[col][0]).sum())
            if n_above or n_below:
                ax.text(
                    0.99, 0.95, f"{n_above} above / {n_below} below view (all works)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray",
                )
        else:
            n_above, n_below = set_robust_ylim(ax, uw_text[col])
        hidden.append(f"{label}: {n_above} above / {n_below} below the visible range (counted over all works)")

        ax.tick_params(axis="x", pad=TICK_PAD)
        for k, fandom in enumerate(text_df["fandom_display"].dropna().unique()):
            annotate_group_counts(
                ax, k,
                lambda cat, f=fandom: ((text_df["fandom_display"] == f) & (text_df["category_clean"] == cat)).sum(),
            )

    use_shared_legend(fig, axes)
    fig.suptitle(f"Word-normalized engagement (text works, \u2265{MIN_WORDS_FOR_NORMALIZATION} words)", y=1.03, fontsize=16)
    if n_excluded > 0:
        fig.text(
            0.5, -0.03,
            f"{n_excluded} very short work(s) (<{MIN_WORDS_FOR_NORMALIZATION} words) excluded: "
            f"per-10k-words ratios are unstable for very short texts.",
            ha="center", fontsize=10, style="italic", color="dimgray",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "03_word_normalized_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '03_word_normalized_boxplots.png'}")

    lines = [
        "WHAT IT SHOWS: kudos, bookmarks and comments per 10,000 words, for text works of at least",
        f"{MIN_WORDS_FOR_NORMALIZATION} words ({n_excluded} shorter text work(s) excluded), for F/F, M/M, F/M and all works.",
        "It asks whether one category earns more engagement per unit of writing.",
        "NUMBERS (median/IQR/n per group):",
    ]
    for col, label in WORD_NORMALIZED_METRICS:
        lines += compare_lines(text_df, col, label, by=["fandom_display"])
    lines.append("POINTS OUTSIDE THE VIEW:")
    lines += ["  " + h for h in hidden]
    lines.append("CAUTION: short works inflate per-word ratios; read alongside figure 06 (length).")
    summary.add("03_word_normalized_boxplots.png - Word-normalized engagement", lines)


# ============================================================
# Plot 4: Rating proxy
# ============================================================

def plot_rating_proxy(df, output_dir, summary):
    rating_order = ["General Audiences", "Teen And Up Audiences", "Mature", "Explicit", "Not Rated"]
    present_ratings = [r for r in rating_order if r in df["rating"].unique()]
    fandoms = df["fandom_display"].dropna().unique()

    fig, axes = plt.subplots(
        1, len(fandoms), figsize=(13 * len(fandoms), 9), squeeze=False, sharey=True
    )
    axes = axes[0]
    for ax, fandom in zip(axes, fandoms):
        fdf = df[(df["fandom_display"] == fandom) & (df["rating"].isin(present_ratings))]
        sns.boxplot(
            data=fdf, x="rating", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER,
            order=present_ratings, palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(fandom, fontsize=15)
        ax.set_ylabel("Kudos / Hits (%)")
        ax.set_xlabel("")
        ax.set_xticks(range(len(present_ratings)))
        ax.set_xticklabels(
            [r.replace(" Audiences", "").replace("Teen And Up", "Teen+") for r in present_ratings],
            fontsize=12,
        )
        ax.tick_params(axis="x", pad=TICK_PAD)

        for k, rating in enumerate(present_ratings):
            annotate_group_counts(
                ax, k,
                lambda cat, r=rating: ((fdf["rating"] == r) & (fdf["category_clean"] == cat)).sum(),
            )

    uw = unique_works(df)
    n_above, n_below = set_robust_ylim(axes[0], uw.loc[uw["rating"].isin(present_ratings), "kudos_to_hits"])
    use_shared_legend(fig, axes)
    fig.suptitle("Kudos/Hits by rating and category, per fandom", y=1.02, fontsize=17)
    fig.text(
        0.5, -0.02,
        "Red n = fewer than 10 works in that box (treat as noise).  " + PROXY_CAVEAT + "\n"
        "Rating measures explicitness, not physical affection.",
        ha="center", va="top", fontsize=11, style="italic", color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "04_rating_proxy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '04_rating_proxy.png'}")

    rated = df[df["rating"].isin(present_ratings)].copy()
    rated["rating"] = pd.Categorical(rated["rating"], categories=present_ratings, ordered=True)
    lines = [
        "WHAT IT SHOWS: kudos/hits split by rating tier for F/F, M/M, F/M and all works, per fandom.",
        "It checks whether a category gap survives within the same explicitness level.",
        "NUMBERS (median kudos/hits, IQR, n):",
    ]
    lines += compare_lines(rated, "kudos_to_hits", "Kudos / Hits (%)", by=["fandom_display", "rating"])
    lines.append(f"Points outside the shared view: {n_above} above / {n_below} below (counted over all works).")
    lines.append("CAUTION: " + PROXY_CAVEAT + " Rating = explicitness, not affection or emotion.")
    summary.add("04_rating_proxy.png - Kudos/hits by rating", lines)


# ============================================================
# Plot 5: Warnings proxy
# ============================================================

def plot_warnings_proxy(df, output_dir, summary):
    df = df.copy()
    df["has_intense_warning"] = df["warnings_clean"].fillna("").str.contains(
        "Major Character Death|Graphic Depictions Of Violence", case=False
    )
    fandoms = df["fandom_display"].dropna().unique()

    fig, axes = plt.subplots(
        1, len(fandoms), figsize=(10 * len(fandoms), 9), squeeze=False, sharey=True
    )
    axes = axes[0]
    for ax, fandom in zip(axes, fandoms):
        fdf = df[df["fandom_display"] == fandom]
        sns.boxplot(
            data=fdf, x="has_intense_warning", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER,
            order=[False, True], palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No death/\nviolence warning", "Death/violence\nwarning"], fontsize=12)
        ax.set_title(fandom, fontsize=15)
        ax.set_ylabel("Kudos / Hits (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", pad=TICK_PAD)

        for k, has_warning in enumerate([False, True]):
            annotate_group_counts(
                ax, k,
                lambda cat, w=has_warning: ((fdf["has_intense_warning"] == w) & (fdf["category_clean"] == cat)).sum(),
            )

    uw = unique_works(df)
    n_above, n_below = set_robust_ylim(axes[0], uw["kudos_to_hits"])
    use_shared_legend(fig, axes)
    fig.suptitle("Kudos/Hits by intense-content warning and category, per fandom", y=1.02, fontsize=16)
    fig.text(
        0.5, -0.02,
        "Red n = fewer than 10 works in that box (treat as noise).  " + PROXY_CAVEAT,
        ha="center", va="top", fontsize=11, style="italic", color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "05_warnings_proxy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '05_warnings_proxy.png'}")

    df["warning_group"] = np.where(df["has_intense_warning"], "death/violence warning", "no death/violence warning")
    lines = [
        "WHAT IT SHOWS: kudos/hits for works with vs without a Major Character Death or Graphic",
        "Depictions of Violence warning, for F/F, M/M, F/M and all works, per fandom. It checks whether",
        "darker content explains a category gap.",
        "NUMBERS (median kudos/hits, IQR, n):",
    ]
    lines += compare_lines(df, "kudos_to_hits", "Kudos / Hits (%)", by=["fandom_display", "warning_group"])
    lines.append(f"Points outside the shared view: {n_above} above / {n_below} below (counted over all works).")
    lines.append("CAUTION: " + PROXY_CAVEAT + " Works with 'Creator Chose Not To Use Archive Warnings'")
    lines.append("are counted as 'no warning' and may contain character death. Rape/Non-Con and Underage")
    lines.append("warnings are not part of this split.")
    summary.add("05_warnings_proxy.png - Kudos/hits by intense-content warning", lines)


# ============================================================
# Plot 6: Length relationship
# ============================================================

def plot_length_relationship(df, output_dir, summary):
    text_df = df[(df["is_text_work"] == True) & (df["words"] > 0)].copy()  # noqa: E712
    text_df["log_words"] = np.log10(text_df["words"])
    all_text = unique_works(text_df)

    fig, ax = plt.subplots(figsize=(13, 8.5))
    n_bins = 8

    # Bin edges come from unique works only, so duplicated 'All works' rows
    # don't count twice.
    shared_bin_edges = None
    if len(all_text) >= n_bins * 6:
        _, shared_bin_edges = pd.qcut(all_text["log_words"], q=n_bins, duplicates="drop", retbins=True)
        shared_bin_edges = np.asarray(shared_bin_edges, dtype=float)
        # Stretch the outer edges so no work in any group falls outside the bins.
        shared_bin_edges[0] = min(shared_bin_edges[0], text_df["log_words"].min())
        shared_bin_edges[-1] = max(shared_bin_edges[-1], text_df["log_words"].max())

    for cat in CATEGORY_ORDER:
        color = CATEGORY_PALETTE[cat]
        subset = text_df[text_df["category_clean"] == cat]
        is_all = cat == ALL_LABEL
        # 'All works' would just repaint every dot, so it gets only its median line.
        if not is_all:
            ax.scatter(subset["log_words"], subset["kudos_to_hits"], alpha=0.2, s=18, color=color,
                       label=f"{cat} (n={len(subset)})")
        else:
            ax.plot([], [], color=color, linestyle="--", linewidth=3, label=f"{display_label(cat)} median line (n={len(subset)})")

        if shared_bin_edges is not None and len(subset) >= n_bins:
            # include_lowest=True keeps the shortest work(s) in the first bin.
            bins = pd.cut(subset["log_words"], bins=shared_bin_edges, include_lowest=True)
            binned = subset.groupby(bins, observed=True).agg(
                x=("log_words", "median"), y=("kudos_to_hits", "median"), n=("log_words", "size")
            ).sort_values("x")
            for _, row in binned.iterrows():
                marker_size = 60 if row["n"] >= 5 else 25
                marker_alpha = 1.0 if row["n"] >= 5 else 0.4
                ax.scatter(row["x"], row["y"], color=color, s=marker_size, alpha=marker_alpha, zorder=5)
            ax.plot(binned["x"], binned["y"], color=color, linewidth=3, alpha=0.85,
                    linestyle="--" if is_all else "-")

    ax.set_xlabel("Word count (log10 scale)")
    ax.set_ylabel("Kudos / Hits (%)")
    ax.set_title("Engagement vs. word count, by category\n(line = median per word-count bin; dashed = all works)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_length_vs_engagement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '06_length_vs_engagement.png'}")

    lines = [
        "WHAT IT SHOWS: each dot is a text work (log10 word count vs kudos/hits) for F/F, M/M and F/M.",
        "Thick lines are the median per word-count bin; all groups, including the dashed 'all works'",
        "line, share the same bin edges, so they are compared at the same lengths. Faded markers = bins",
        "with fewer than 5 works.",
        "NUMBERS:",
    ]
    for cat in CATEGORY_ORDER:
        s = text_df[text_df["category_clean"] == cat].dropna(subset=["log_words", "kudos_to_hits"])
        if len(s) >= 3:
            rho, p = stats.spearmanr(s["log_words"], s["kudos_to_hits"])
            direction = "longer works have higher kudos/hits" if rho > 0 else "longer works have lower kudos/hits"
            # The weighted all-works group is a resample (repeated works), so its p-value would mislead.
            p_txt = "" if cat == ALL_LABEL else f" (p={p:.3g})"
            lines.append(
                f"  {display_label(cat)}: n={len(s)}, median words {s['words'].median():,.0f}, "
                f"Spearman rho(length, kudos/hits)={rho:.2f}{p_txt} -> {direction}"
            )
        else:
            lines.append(f"  {cat}: n={len(s)} (too few for a correlation)")
    if shared_bin_edges is None:
        lines.append("  Not enough works for binned median lines (need >= 48 text works).")
    lines.append("CAUTION: if the groups have different length distributions, a raw category gap may partly")
    lines.append("be a length effect. Compare the lines at the same x position.")
    summary.add("06_length_vs_engagement.png - Engagement vs word count", lines)


# ============================================================
# Plot 7: Supply / discovery / demand
# ============================================================

def plot_supply_discovery_demand(df, output_dir, summary):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    counts = df.groupby(["fandom_display", "category_clean"]).size().reset_index(name="count")
    sns.barplot(data=counts, x="fandom_display", y="count", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[0])
    axes[0].set_title("Supply: works in YOUR sample")
    axes[0].set_ylabel("Number of works")
    axes[0].set_xlabel("")

    sns.boxplot(data=df, x="fandom_display", y="hits", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[1], showfliers=True, flierprops=FLIER_PROPS)
    axes[1].set_title("Discovery: hits")
    axes[1].set_ylabel("Hits")
    axes[1].set_xlabel("")
    axes[1].set_yscale("log")

    sns.boxplot(data=df, x="fandom_display", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[2], showfliers=True, flierprops=FLIER_PROPS)
    axes[2].set_title("Demand given discovery: kudos/hits")
    axes[2].set_ylabel("Kudos / Hits (%)")
    axes[2].set_xlabel("")

    # One shared legend instead of three.
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles=category_legend_handles(), title="Category", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=True)

    fig.suptitle("Three different bottlenecks, not one 'popularity'", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "07_supply_discovery_demand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '07_supply_discovery_demand.png'}")

    lines = [
        "WHAT IT SHOWS: three separate quantities instead of one 'popularity', for F/F, M/M, F/M and all works.",
        "  Supply = number of works in the sample. Discovery = hits (log scale). Demand given",
        "  discovery = kudos/hits.",
        "SUPPLY (works in sample):",
    ]
    for _, r in counts.iterrows():
        lines.append(f"  [{r['fandom_display']} / {r['category_clean']}] {int(r['count'])} works")
    lines.append("DISCOVERY / DEMAND (median/IQR/n per group):")
    lines += compare_lines(df, "hits", "Hits", by=["fandom_display"])
    lines += compare_lines(df, "kudos_to_hits", "Kudos / Hits (%)", by=["fandom_display"])
    lines.append("CAUTION: pages were sampled randomly, but HOW MANY pages you scraped per cohort sets these counts,")
    lines.append("so the supply panel is not evidence about real supply on AO3. 'All works' counts add up the")
    lines.append("other cohorts plus Gen/Multi/Other works. Hits and demand are the informative panels.")
    summary.add("07_supply_discovery_demand.png - Supply / discovery / demand", lines)


# ============================================================
# Supplementary: Mann-Whitney U tests
# ============================================================

def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def run_group_comparisons(df, summary):
    header = [
        "=" * 60,
        "MANN-WHITNEY U COMPARISONS BETWEEN CATEGORIES, BY FANDOM",
        "DESCRIPTIVE / HYPOTHESIS-GENERATING ONLY",
        "=" * 60,
    ]
    for h in header:
        print(h)

    # Tests use the SAME rows as the plots: hit-based ratios on all works,
    # word-normalized ratios on text works >= MIN_WORDS_FOR_NORMALIZATION.
    # 'All works' is not tested: it contains the other groups.
    text_df = text_work_subset(df)
    plan = [(col, label, df) for col, label in ENGAGEMENT_METRICS] + \
           [(col, label, text_df) for col, label in WORD_NORMALIZED_METRICS]
    pairs = [(a, b) for a, b in GAP_PAIRS if a in CATEGORY_ORDER and b in CATEGORY_ORDER]

    results = []
    for fandom in df["fandom_display"].dropna().unique():
        for a, b in pairs:
            for col, label, source in plan:
                fdf = source[source["fandom_display"] == fandom]
                ra = fdf[fdf["category_clean"] == a]
                rb = fdf[fdf["category_clean"] == b]
                # A work tagged with BOTH categories would sit in both samples, which the test
                # does not allow (the groups must not share works), so it is left out of this pair.
                shared = set(ra["work_id"]) & set(rb["work_id"])
                va = ra.loc[~ra["work_id"].isin(shared), col].dropna()
                vb = rb.loc[~rb["work_id"].isin(shared), col].dropna()
                base = {"fandom": fandom, "pair": (a, b), "label": label,
                        "n_a": len(va), "n_b": len(vb), "n_shared": len(shared)}
                if len(va) < MIN_GROUP_N_FOR_TEST or len(vb) < MIN_GROUP_N_FOR_TEST:
                    results.append({**base, "skipped": True})
                    continue
                u, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
                delta = 2 * u / (len(va) * len(vb)) - 1  # Cliff's delta: >0 means A tends higher
                results.append({**base, "skipped": False, "p": p, "delta": delta,
                                "median_a": va.median(), "median_b": vb.median()})

    tested = [r for r in results if not r["skipped"]]
    if tested:
        adjusted = benjamini_hochberg([r["p"] for r in tested])
        for r, p_adj in zip(tested, adjusted):
            r["p_adj"] = p_adj

    lines = [
        "WHAT IT SHOWS: for each fandom, pair of categories and metric, whether the two distributions",
        "differ (Mann-Whitney U, two-sided; Benjamini-Hochberg FDR across ALL tests run). Pairs tested:",
        "  " + "; ".join(f"{a} vs {b}" for a, b in pairs) + ". 'All works' is not tested (it contains the others).",
        "Works tagged with BOTH categories of a pair are left out of that pair's test so the two groups",
        "share no works (the number left out is shown as 'overlap excluded').",
        "Cliff's delta is the effect size: -1 to +1, positive = the FIRST group's values tend to be higher;",
        "roughly |d|<0.15 negligible, <0.33 small, <0.47 medium, otherwise large. ** = FDR-adjusted p < 0.05.",
        f"Groups need n >= {MIN_GROUP_N_FOR_TEST} in BOTH categories or the test is skipped.",
    ]
    for fandom in df["fandom_display"].dropna().unique():
        print(f"\n--- {fandom} ---")
        lines.append(f"[{fandom}]")
        for r in results:
            if r["fandom"] != fandom:
                continue
            a, b = r["pair"]
            if r["skipped"]:
                txt = (f"  {a} vs {b} | {r['label']:26s}: skipped (n too small: {a}={r['n_a']}, "
                       f"{b}={r['n_b']}, need >={MIN_GROUP_N_FOR_TEST})")
            else:
                flag = " **" if r["p_adj"] < 0.05 else ""
                txt = (f"  {a} vs {b} | {r['label']:26s}: {a} median={r['median_a']:.2f}, "
                       f"{b} median={r['median_b']:.2f}, Cliff's d={r['delta']:+.2f}, "
                       f"raw p={r['p']:.4f}, FDR p={r['p_adj']:.4f}{flag} "
                       f"(n_{a}={r['n_a']}, n_{b}={r['n_b']}, overlap excluded={r['n_shared']})")
            print(txt)
            lines.append(txt)
    lines.append("CAUTION: works by the same author are not independent, so p-values are optimistic. A small")
    lines.append("p with a negligible Cliff's d is a detectable but unimportant difference.")
    summary.add("Statistical tests (printed to terminal; not a figure)", lines)


# ============================================================
# Plot 8: sample vs true supply
# ============================================================

SUPPLY_CATEGORIES = ["F/F", "M/M", "F/M", "Gen", "Multi", "Other"]


def plot_sample_vs_supply(table, meta, supply, output_dir, summary):
    tdf = pd.DataFrame(table)
    fandoms = list(meta.keys())
    fig, axes = plt.subplots(1, len(fandoms), figsize=(8.5 * len(fandoms), 6.5), squeeze=False, sharey=True)
    axes = axes[0]
    for ax, fandom in zip(axes, fandoms):
        f = tdf[tdf["fandom"] == fandom].set_index("category").reindex(SUPPLY_CATEGORIES)
        x = np.arange(len(SUPPLY_CATEGORIES))
        w = 0.27
        for offset, col, label, color in [
            (-w, "true_share", "True AO3 share", "#333333"),
            (0, "sample_share", "Your sample (unweighted)", "#E69F00"),
            (w, "weighted_share", "Your sample after weighting", "#009E73"),
        ]:
            ax.bar(x + offset, f[col] * 100, w, label=label, color=color)
            for xi, v in enumerate(f[col]):
                ax.text(xi + offset, v * 100 + 0.6, f"{v * 100:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(SUPPLY_CATEGORIES, fontsize=11)
        ax.set_title(fandom)
        ax.set_ylabel("% of works listing it", fontsize=13)
        ax.legend(fontsize=9)
    fig.suptitle("Share of works listing each category: real AO3 vs your sample (bars can add to >100%: works list several)",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "08_sample_vs_true_supply.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '08_sample_vs_true_supply.png'}")

    flt = supply["filters"]
    lines = [
        "WHAT IT SHOWS: for each fandom, the share of works that LIST each category on AO3 itself (dark),",
        "in your sample's all-works set (orange), and in the weighted all-works set used in the other",
        "figures (green). A work that lists two categories counts in both, so shares add to more than 100%.",
        "Where dark and orange differ, your sample over- or under-represents that category. Weights are",
        "found by raking: each work's weight is adjusted until the weighted share of works listing each",
        "category matches AO3's. F/F, M/M and F/M groups are not changed by the weighting.",
        f"HOW THE TRUE COUNTS WERE OBTAINED: AO3 search with crossover={flt['crossover']}, warnings={flt['warning']},",
        f"completion={flt['completion']}, timeframe={flt['timeframe']}"
        + (", hits >= 50 (same floor as the scraper)." if supply["hits_floor"] else ", no hits floor."),
        "(Fandoms whose source says 'manual' use your own counts, which have NO hits floor.)",
        "NUMBERS:",
    ]
    for fandom, m in meta.items():
        lines.append(f"[{fandom}] source: {m['provenance']}")
        lines.append(f"  all-works count used: {m['all_count']:,} ({m['all_count_source']})")
        lines.append(
            f"  effective sample size after weighting: {m['ess']:,.0f} of {m['n']:,} works "
            f"(smaller = the weights lean on fewer works, so the weighted line is less precise)."
        )
        lines.append(
            f"  largest remaining miss between weighted and true share: {m['max_err'] * 100:.2f} points"
            + (f"; {m['n_capped']} weights were capped" if m["n_capped"] else "")
        )
        for r in table:
            if r["fandom"] != fandom:
                continue
            lines.append(
                f"  {r['category']:>5}: true {r['true_count']:,} ({r['true_share']:.1%}) | "
                f"sample {r['sample_n']:,} ({r['sample_share']:.1%}) | after weighting {r['weighted_share']:.1%} | "
                f"you sampled {r['rate']:.1%} of them | average weight x{r['avg_weight']:.2f}"
            )
    lines.append("CAUTION: Gen/Multi/Other are only as representative as the way those works entered your CSV")
    lines.append("(they arrive as by-products of other scrapes unless you scraped with the category filter off).")
    lines.append("A low sampling rate in one category (often the hits floor removing low-traffic works) shows")
    lines.append("up here as a bar mismatch.")
    summary.add("08_sample_vs_true_supply.png - Sample mix vs real AO3 mix", lines)


# ============================================================
# Deeper analyses (figures 09-11): single-category works, joint models, ships
# ============================================================

try:
    import statsmodels.api as sm
except ImportError:  # the model section reports this instead of crashing
    sm = None

DEEP_METRICS = [("bookmarks_to_hits", "Bookmarks / Hits (%)"), ("kudos_to_hits", "Kudos / Hits (%)")]
CONTRASTS = [("M/M", "F/F"), ("M/M", "F/M"), ("F/F", "F/M")]
CONTRAST_COLORS = {"M/M vs F/F": "#1B9E77", "M/M vs F/M": "#7570B3", "F/F vs F/M": "#D95F02"}
MIN_SHIP_WORKS = 10          # a ship needs this many sampled works to appear as a point
CANON_FILE = "ship_canon_labels.csv"
LEAVE_OUT_STEPS = (0, 1, 3, 5, 10)


def cliffs_delta(a, b):
    """Cliff's delta (>0: a tends higher than b) and the Mann-Whitney p-value."""
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1, p


def lighten(hex_color, amount=0.55):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(*(int(c + (255 - c) * amount) for c in (r, g, b)))


# ------------------------------------------------------------
# Item 4: works listing ONE category vs works listing several
# ------------------------------------------------------------

def plot_pure_vs_mixed(df, output_dir, summary):
    from matplotlib.patches import Patch

    d = df[df["category_clean"].isin(PURE_CATEGORIES)].copy()
    if d.empty:
        return
    d["subgroup"] = d["category_clean"] + np.where(d["n_categories"] == 1, " only", " + others")
    wanted = [f"{c} only" for c in PURE_CATEGORIES] + [f"{c} + others" for c in PURE_CATEGORIES]
    order = [s for s in wanted if s in set(d["subgroup"])]
    palette = {s: (CATEGORY_PALETTE[s.split(" ")[0]] if s.endswith("only")
                   else lighten(CATEGORY_PALETTE[s.split(" ")[0]])) for s in order}
    fandoms = list(d["fandom_display"].dropna().unique())
    width = 0.8 / len(order)
    offsets = {s: -0.4 + width * (i + 0.5) for i, s in enumerate(order)}
    unique_d = d.drop_duplicates("work_id")

    fig, axes = plt.subplots(len(DEEP_METRICS), 1, figsize=(17, 7 * len(DEEP_METRICS)))
    for ax, (col, label) in zip(axes, DEEP_METRICS):
        sns.boxplot(data=d, x="fandom_display", y=col, hue="subgroup", hue_order=order, palette=palette,
                    ax=ax, showfliers=True, flierprops=FLIER_PROPS)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        set_robust_ylim(ax, unique_d[col])
        ax.tick_params(axis="x", pad=22)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for k, f in enumerate(fandoms):
            for s in order:
                n = int(((d["fandom_display"] == f) & (d["subgroup"] == s)).sum())
                annotate_n(ax, k + offsets[s], n)
    fig.legend(handles=[Patch(facecolor=palette[s], edgecolor="black", label=s) for s in order],
               title="Works that list...", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle("Works listing ONLY one category (dark) vs that category plus others (light)", y=1.0, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "09_single_vs_multi_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '09_single_vs_multi_category.png'}")

    lines = [
        "WHAT IT SHOWS: each category split into works that list ONLY that category (dark boxes) and works that",
        "list it together with other categories (light boxes). A work tagged F/F and M/M appears in both light",
        "groups. The dark groups share no works, so they are the cleanest category comparison available.",
        "SHARE OF EACH CATEGORY THAT IS MULTI-CATEGORY:",
    ]
    for f in fandoms:
        for c in PURE_CATEGORIES:
            sub = d[(d["fandom_display"] == f) & (d["category_clean"] == c)]
            if len(sub):
                n_mixed = int((sub["n_categories"] > 1).sum())
                lines.append(f"  [{f}] {c}: {len(sub) - n_mixed:,} list only {c}; {n_mixed:,} also list others "
                             f"({n_mixed / len(sub):.0%})")
    lines.append("NUMBERS (median/IQR/n):")
    for col, label in DEEP_METRICS:
        lines.append(f"  {label}:")
        for f in fandoms:
            lines.append(f"    [{f}]")
            for s in order:
                v = d.loc[(d["fandom_display"] == f) & (d["subgroup"] == s), col].dropna()
                if len(v):
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    lines.append(f"      {s}: median {med:.2f} (IQR {q1:.2f}-{q3:.2f}, n={len(v)})")

    tests = []
    for f in fandoms:
        fd = d[d["fandom_display"] == f]
        for col, label in DEEP_METRICS:
            groups = {s: fd.loc[fd["subgroup"] == s, col].dropna() for s in order}
            pairs = [(f"{a} only", f"{b} only") for a, b in CONTRASTS] + \
                    [(f"{c} only", f"{c} + others") for c in PURE_CATEGORIES]
            for x, y in pairs:
                if x in groups and y in groups and min(len(groups[x]), len(groups[y])) >= MIN_GROUP_N_FOR_TEST:
                    delta, p = cliffs_delta(groups[x], groups[y])
                    tests.append({"fandom": f, "label": label, "x": x, "y": y, "delta": delta, "p": p,
                                  "mx": groups[x].median(), "my": groups[y].median(),
                                  "nx": len(groups[x]), "ny": len(groups[y])})
    if tests:
        for t, p_adj in zip(tests, benjamini_hochberg([t["p"] for t in tests])):
            t["p_adj"] = p_adj
    lines.append("COMPARISONS (Cliff's delta: positive = first group higher; ** = FDR-adjusted p < 0.05):")
    for f in fandoms:
        lines.append(f"  [{f}]")
        for t in tests:
            if t["fandom"] == f:
                flag = " **" if t["p_adj"] < 0.05 else ""
                lines.append(f"    {t['x']} vs {t['y']} | {t['label']}: {t['mx']:.2f} vs {t['my']:.2f}, "
                             f"d={t['delta']:+.2f}, FDR p={t['p_adj']:.4f}{flag} (n={t['nx']}, {t['ny']})")
    lines.append("CAUTION: 'only' vs '+ others' compares works that differ in more than the tag (multi-category works")
    lines.append("tend to be longer or broader), so a gap here is a flag to investigate, not a finding about tagging.")
    summary.add("09_single_vs_multi_category.png - Works listing one category vs several", lines)


# ------------------------------------------------------------
# Ships (used by the models and by figure 11)
# ------------------------------------------------------------

def primary_ship(text):
    """First romantic ('/') relationship listed, name order ignored."""
    if not isinstance(text, str):
        return None
    for token in text.split(","):
        if "/" in token:
            parts = [p.strip() for p in token.split("/") if p.strip()]
            if len(parts) >= 2:
                return " / ".join(sorted(parts))
    return None


def add_ship_columns(uw):
    uw = uw.copy()
    uw["primary_ship"] = uw["relationships"].map(primary_ship) if "relationships" in uw.columns else None
    key = uw["fandom_display"].astype(str) + "|" + uw["primary_ship"].astype(str)
    sizes = uw.dropna(subset=["primary_ship"]).assign(_k=key).groupby("_k").size()
    uw["ship_n"] = key.map(sizes)
    return uw


def ship_table(uw):
    s = uw.dropna(subset=["primary_ship", "fandom_display"])
    if s.empty:
        return pd.DataFrame()
    g = s.groupby(["fandom_display", "primary_ship"]).agg(
        n_works=("work_id", "size"), bm_median=("bookmarks_to_hits", "median"),
        kudos_median=("kudos_to_hits", "median"),
        ff=("has_F/F", "sum"), mm=("has_M/M", "sum"), fm=("has_F/M", "sum"),
    ).reset_index()
    best = g[["ff", "mm", "fm"]].idxmax(axis=1).map({"ff": "F/F", "mm": "M/M", "fm": "F/M"})
    g["group"] = np.where(g[["ff", "mm", "fm"]].max(axis=1) > 0, best, "other")
    return g


def load_canon_labels():
    if not Path(CANON_FILE).exists():
        return None
    c = pd.read_csv(CANON_FILE, dtype=str).fillna("")
    c["canon"] = c["canon"].str.strip().str.lower()
    c = c[c["canon"].isin(["canon", "implied", "non-canon"])]
    return c if len(c) else None


def attach_canon(uw, canon):
    mapping = {(r["fandom"], r["ship"]): r["canon"] for _, r in canon.iterrows()}
    uw = uw.copy()
    uw["canon_label"] = [mapping.get((f, s), "unlabelled") for f, s in zip(uw["fandom_display"], uw["primary_ship"])]
    return uw


def write_canon_template(ships, per_fandom=25):
    """One-off helper: a CSV of the biggest ships for you to label canon / implied / non-canon."""
    if ships.empty or Path(CANON_FILE).exists():
        return
    rows = []
    for f, g in ships.groupby("fandom_display"):
        top = g.sort_values("n_works", ascending=False).head(per_fandom)
        rows.append(pd.DataFrame({"fandom": f, "ship": top["primary_ship"], "group": top["group"],
                                  "n_works": top["n_works"], "canon": ""}))
    pd.concat(rows).to_csv(CANON_FILE, index=False)
    print(f"\nWrote {CANON_FILE}: fill its 'canon' column with canon / implied / non-canon, then re-run.")


# ------------------------------------------------------------
# Item 5: joint model (everything at once)
# ------------------------------------------------------------

def _model_frame(uw):
    d = uw[(uw["hits"] > 0) & uw["is_text_work"] & (uw["words"] > 0)].dropna(subset=["fandom_display", "rating"]).copy()
    chapters = pd.to_numeric(d["chapters"], errors="coerce") if "chapters" in d.columns else pd.Series(1, index=d.index)
    d["chapters"] = chapters.fillna(1).clip(lower=1)
    for c in CATEGORY_NAMES:
        d[f"cat_{c}"] = d[f"has_{c}"].astype(float)
    wc = d["warnings_clean"].fillna("")
    d["warn_intense"] = wc.str.contains("Major Character Death|Graphic Depictions Of Violence", case=False).astype(float)
    d["warn_other"] = wc.str.contains("Rape/Non-Con|Underage", case=False).astype(float)
    d["warn_unchosen"] = wc.str.contains("Creator Chose Not To Use", case=False).astype(float)

    def z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    d["z_log_words"] = z(np.log10(d["words"]))
    d["z_log_hits"] = z(np.log10(d["hits"]))
    d["z_log_chapters"] = z(np.log10(d["chapters"]))
    has_ship = d["ship_n"].notna()
    d["no_ship"] = (~has_ship).astype(float)
    log_ship = np.log10(d["ship_n"].where(has_ship, 1))
    d["z_log_ship_size"] = z(log_ship.where(has_ship)).fillna(0.0)
    return d


DROPPED_TERMS = set()


def _drop_redundant(X):
    """Keep columns in order, skipping any that is an exact combination of the ones before it
    (e.g. 'Gen' and 'no ship' being the same works). Earlier columns win, and the category terms
    come first, so they are never the ones dropped. Dropped names are reported in the summary."""
    values = X.to_numpy(float)
    kept = []
    for j, name in enumerate(X.columns):
        trial = kept + [j]
        if np.linalg.matrix_rank(values[:, trial]) == len(trial):
            kept = trial
        else:
            DROPPED_TERMS.add(name)
    return X.iloc[:, kept]


def _design(d, pooled, spec_b, use_canon):
    cols = [f"cat_{c}" for c in CATEGORY_NAMES] + [
        "warn_intense", "warn_other", "warn_unchosen", "z_log_words", "z_log_hits", "z_log_chapters"]
    X = d[cols].copy()
    rating = pd.get_dummies(d["rating"], prefix="rating").astype(float)
    ref = "rating_General Audiences" if "rating_General Audiences" in rating.columns else rating.columns[0]
    X = pd.concat([X, rating.drop(columns=[ref])], axis=1)
    if pooled:
        fd = pd.get_dummies(d["fandom_display"], prefix="fandom").astype(float)
        X = pd.concat([X, fd.iloc[:, 1:]], axis=1)
    if spec_b:
        X["z_log_ship_size"] = d["z_log_ship_size"]
        X["no_ship"] = d["no_ship"]
        if use_canon:
            for lab in ("canon", "implied", "unlabelled"):
                X[f"canon_{lab}"] = (d["canon_label"] == lab).astype(float)
    X = X.loc[:, X.std() > 0]
    return _drop_redundant(sm.add_constant(X, has_constant="add"))


def _fit_glm(d, outcome, pooled, spec_b, use_canon):
    X = _design(d, pooled, spec_b, use_canon)
    hits = d["hits"].to_numpy(float)
    y = np.minimum(d[outcome].to_numpy(float), hits) / hits
    model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=hits)
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        return model.fit(cov_type="HC1")


def _contrast(res, a, b):
    na, nb = f"cat_{a}", f"cat_{b}"
    if na not in res.params.index or nb not in res.params.index:
        return None
    cov = res.cov_params()
    diff = res.params[na] - res.params[nb]
    se = float(np.sqrt(cov.loc[na, na] + cov.loc[nb, nb] - 2 * cov.loc[na, nb]))
    return {"diff": float(diff), "se": se, "rr": float(np.exp(diff)),
            "lo": float(np.exp(diff - 1.96 * se)), "hi": float(np.exp(diff + 1.96 * se))}


def run_models(uw, has_canon, output_dir, summary):
    title = "10_joint_model_effects.png - Category effects with everything controlled at once"
    if sm is None:
        summary.add(title, ["SKIPPED: statsmodels is not installed (pip3 install statsmodels)."])
        print("Skipping models: pip install statsmodels")
        return
    d = _model_frame(uw)
    fandoms = list(d["fandom_display"].unique())
    scopes = ["All fandoms"] + fandoms
    outcomes = [("bookmarks", "Bookmarks per visit"), ("kudos", "Kudos per visit")]
    specs = [("A", False), ("B", True)]

    results, pooled_fit = [], {}
    for outcome, _ in outcomes:
        for spec, spec_b in specs:
            for scope in scopes:
                sub = d if scope == "All fandoms" else d[d["fandom_display"] == scope]
                if len(sub) < 200:
                    continue
                res = _fit_glm(sub, outcome, scope == "All fandoms", spec_b, has_canon)
                if scope == "All fandoms":
                    pooled_fit[(outcome, spec)] = res
                for a, b in CONTRASTS:
                    ct = _contrast(res, a, b)
                    if ct:
                        results.append({"outcome": outcome, "spec": spec, "scope": scope, "a": a, "b": b,
                                        "n": int(res.nobs), **ct})

    rows = [(s, a, b) for s in scopes for a, b in CONTRASTS]
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(18, 0.5 * len(rows) + 3), sharey=True)
    for ax, (outcome, ttl) in zip(axes, outcomes):
        for r in results:
            if r["outcome"] != outcome:
                continue
            color = CONTRAST_COLORS[f"{r['a']} vs {r['b']}"]
            y = ypos[(r["scope"], r["a"], r["b"])] + (0.14 if r["spec"] == "A" else -0.14)
            ax.errorbar(r["rr"], y, xerr=[[r["rr"] - r["lo"]], [r["hi"] - r["rr"]]],
                        fmt="o" if r["spec"] == "A" else "s", color=color,
                        mfc="white" if r["spec"] == "A" else color, capsize=2, markersize=6)
        ax.axvline(1, color="black", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(f"{ttl}\n(rate ratio, 95% interval; right of 1 = first group converts more visits)", fontsize=12)
        ax.set_xlabel("Rate ratio (log scale)")
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([f"{s.rstrip(':')}: {a} vs {b}" for s, a, b in rows], fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=3, label=k) for k, c in CONTRAST_COLORS.items()] + [
        Line2D([0], [0], marker="o", color="gray", mfc="white", lw=0, label="A: length, hits, rating, warnings"),
        Line2D([0], [0], marker="s", color="gray", lw=0,
               label="B: A + ship popularity" + (" + canon" if has_canon else ""))]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle("Category effects after controlling for length, hits, rating, warnings (A) and ship (B)", y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "10_joint_model_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '10_joint_model_effects.png'}")

    lines = [
        "WHAT IT SHOWS: the M/M vs F/F, M/M vs F/M and F/F vs F/M gaps in how often a VISIT turns into a bookmark",
        "(or kudos), after controlling for everything at once: fandom, rating, warnings, length, number of chapters,",
        "how many hits the work has (casual traffic) and, in model B, how big the work's ship is in your sample"
        + (" and its canon label." if has_canon else ". (No canon labels filled in yet.)"),
        "METHOD: binomial regression with each visit as a trial (a standard way to model 'X out of N visits'),",
        "with standard errors that do not assume the textbook variance (HC1). A rate ratio of 1.30 means a visit",
        "to the first group's work is about 30% more likely to end in a bookmark than a visit to the second's,",
        "all else equal. A work listing several categories counts toward each of them. 95% intervals that",
        "exclude 1 mean the gap is distinguishable from zero. Open circles = model A, filled squares = model B.",
        f"WORKS USED: {len(d):,} text works with hits>0 (works with unusable length data are left out).",
    ]
    for outcome, ttl in outcomes:
        for spec, _ in specs:
            sel = [r for r in results if r["outcome"] == outcome and r["spec"] == spec]
            if not sel:
                continue
            lines.append(f"{ttl.upper()} - model {spec}:")
            for scope in scopes:
                parts = [f"{r['a']} vs {r['b']}: x{r['rr']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})"
                         for r in sel if r["scope"] == scope]
                if parts:
                    n = next(r["n"] for r in sel if r["scope"] == scope)
                    lines.append(f"  [{scope.rstrip(':')}, n={n:,}] " + "; ".join(parts))
    lines.append("DO THE FANDOMS DIFFER? (difference in the M/M vs F/F effect between two fandoms, log scale;")
    lines.append("a small p means the M/M advantage genuinely differs between those fandoms):")
    for outcome, ttl in outcomes:
        for spec, _ in specs:
            sel = {r["scope"]: r for r in results if r["outcome"] == outcome and r["spec"] == spec
                   and (r["a"], r["b"]) == ("M/M", "F/F") and r["scope"] != "All fandoms"}
            names = list(sel)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    r1, r2 = sel[names[i]], sel[names[j]]
                    z = (r1["diff"] - r2["diff"]) / np.sqrt(r1["se"] ** 2 + r2["se"] ** 2)
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                    lines.append(f"  {ttl}, model {spec}: {names[i].rstrip(':')} vs {names[j].rstrip(':')}: z={z:+.2f}, p={p:.4f}")
    res = pooled_fit.get(("bookmarks", "A"))
    if res is not None:
        lines.append("ALL OTHER TERMS (bookmarks, model A, all fandoms): rate ratio per term. Continuous terms are")
        lines.append("per +1 standard deviation; category terms compare against works listing NO category.")
        conf = res.conf_int()
        for name in res.params.index:
            if name == "const":
                continue
            lines.append(f"  {name}: x{np.exp(res.params[name]):.2f} "
                         f"({np.exp(conf.loc[name, 0]):.2f}-{np.exp(conf.loc[name, 1]):.2f})")
        explained = 1 - res.deviance / res.null_deviance if res.null_deviance else float("nan")
        lines.append(f"  Share of the variation in bookmark rate explained by model A: {explained:.1%}")
    if DROPPED_TERMS:
        lines.append("TERMS DROPPED AS REDUNDANT (identical to a combination of other terms in at least one model): "
                     + ", ".join(sorted(DROPPED_TERMS)))
    lines.append("CAUTION: association, not cause. Works by the same author are not independent (no author column),")
    lines.append("so intervals are somewhat too narrow. Ship popularity is measured from YOUR sample, and canon")
    lines.append("status only enters if you fill in ship_canon_labels.csv.")
    summary.add(title, lines)


# ------------------------------------------------------------
# Item 6: ships and canon
# ------------------------------------------------------------

def _leave_out(uw, fandom, outcome, a, b):
    fd = uw[uw["fandom_display"] == fandom]
    A = fd[fd[f"has_{a}"] & ~fd[f"has_{b}"]]
    B = fd[fd[f"has_{b}"] & ~fd[f"has_{a}"]]
    top_a = A["primary_ship"].value_counts().index.tolist()
    top_b = B["primary_ship"].value_counts().index.tolist()
    out = []
    for k in LEAVE_OUT_STEPS:
        A2 = A[~A["primary_ship"].isin(top_a[:k])] if k else A
        B2 = B[~B["primary_ship"].isin(top_b[:k])] if k else B
        va, vb = A2[outcome].dropna(), B2[outcome].dropna()
        if min(len(va), len(vb)) >= MIN_GROUP_N_FOR_TEST:
            delta, _ = cliffs_delta(va, vb)
            out.append((k, delta, len(va), len(vb)))
    return out


def plot_ship_analyses(uw, ships, has_canon, output_dir, summary):
    title = "11_ship_analysis.png - Ships: concentration, leave-top-ships-out, ship-level view"
    if ships.empty:
        summary.add(title, ["SKIPPED: no romantic ('/') relationships found in the relationships column."])
        return
    fandoms = list(uw["fandom_display"].dropna().unique())
    fig, axes = plt.subplots(3, len(fandoms), figsize=(7 * len(fandoms), 17), squeeze=False)
    conc_lines, leave_lines, ship_lines = [], [], []
    canon_markers = {"canon": "o", "implied": "s", "non-canon": "X", "unlabelled": "o"}

    for j, f in enumerate(fandoms):
        fd = uw[uw["fandom_display"] == f]

        # Row 1: how concentrated each category is in a few ships
        ax = axes[0][j]
        top1, top5, ticks = [], [], []
        conc_lines.append(f"  [{f}]")
        for c in PURE_CATEGORIES:
            members = fd[fd[f"has_{c}"]]
            with_ship = members.dropna(subset=["primary_ship"])
            counts = with_ship["primary_ship"].value_counts()
            if counts.empty:
                top1.append(0)
                top5.append(0)
                ticks.append(f"{c}\n(no ships)")
                continue
            top1.append(counts.iloc[0] / counts.sum() * 100)
            top5.append(counts.head(5).sum() / counts.sum() * 100)
            ticks.append(f"{c}\n(ship listed: {len(with_ship) / len(members):.0%})")
            conc_lines.append(
                f"    {c}: top ship '{counts.index[0]}' = {top1[-1]:.0f}% of works with a ship; top 5 = {top5[-1]:.0f}%. "
                f"Top 3: " + "; ".join(f"{s} ({n})" for s, n in counts.head(3).items()))
        x = np.arange(len(PURE_CATEGORIES))
        ax.bar(x - 0.2, top1, 0.4, label="Biggest ship", color="#444444")
        ax.bar(x + 0.2, top5, 0.4, label="Top 5 ships", color="#999999")
        for xi, (a1, a5) in enumerate(zip(top1, top5)):
            ax.text(xi - 0.2, a1 + 1, f"{a1:.0f}", ha="center", fontsize=9)
            ax.text(xi + 0.2, a5 + 1, f"{a5:.0f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=10)
        ax.set_ylim(0, 118)
        ax.set_title(f"{f.rstrip(':')}\nShare of works held by the biggest ships", fontsize=13)
        ax.set_ylabel("% of works that list a ship", fontsize=11)
        if j == 0:
            ax.legend(loc="upper center", ncol=2, fontsize=9)

        # Row 2: does the M/M gap survive removing each group's biggest ships?
        ax = axes[1][j]
        leave_lines.append(f"  [{f}]")
        for (a, b), style in zip(CONTRASTS[:2], ["-", "--"]):
            for outcome, olabel in DEEP_METRICS:
                res = _leave_out(uw, f, outcome, a, b)
                if not res:
                    continue
                if outcome == "bookmarks_to_hits":
                    ax.plot([r[0] for r in res], [r[1] for r in res], style, marker="o", linewidth=2.5,
                            color=CONTRAST_COLORS[f"{a} vs {b}"], label=f"{a} vs {b}")
                leave_lines.append(f"    {a} vs {b} | {olabel}: " + "; ".join(
                    f"top {k} removed d={dl:+.2f} (n={na},{nb})" for k, dl, na, nb in res))
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(list(LEAVE_OUT_STEPS))
        ax.set_xlabel("Biggest ships removed from each group")
        ax.set_ylabel("Cliff's delta, bookmarks per hit", fontsize=11)
        ax.set_title("Is the gap just a few mega-ships?")
        ax.legend()

        # Row 3: every ship as a point
        ax = axes[2][j]
        sh = ships[(ships["fandom_display"] == f) & (ships["n_works"] >= MIN_SHIP_WORKS)].copy()
        if has_canon and "canon_label" in uw.columns:
            lab = uw.dropna(subset=["primary_ship"]).groupby(["fandom_display", "primary_ship"])["canon_label"].agg(
                lambda s: s.mode().iloc[0])
            sh["canon_label"] = [lab.get((f, s), "unlabelled") for s in sh["primary_ship"]]
        else:
            sh["canon_label"] = "unlabelled"
        for cat, color in list(CATEGORY_PALETTE.items())[:3] + [("other", "#BBBBBB")]:
            for lbl, mk in canon_markers.items():
                part = sh[(sh["group"] == cat) & (sh["canon_label"] == lbl)]
                if part.empty:
                    continue
                ax.scatter(np.log10(part["n_works"]), part["bm_median"], s=55, color=color, marker=mk, alpha=0.75,
                           edgecolor="black", linewidth=0.4,
                           label=f"{cat}" + (f", {lbl}" if has_canon else ""))
        for _, r in sh.sort_values("n_works", ascending=False).head(4).iterrows():
            ax.annotate(r["primary_ship"][:30], (np.log10(r["n_works"]), r["bm_median"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Ship size: works in your sample (log10)")
        ax.set_ylabel("Ship's median bookmarks per hit (%)", fontsize=11)
        ax.set_title(f"Each point = one ship with at least {MIN_SHIP_WORKS} works", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8)

        ship_lines.append(f"  [{f}] ships with >= {MIN_SHIP_WORKS} works, one value per ship:")
        groups = {c: sh.loc[sh["group"] == c, "bm_median"].dropna() for c in PURE_CATEGORIES}
        for c, v in groups.items():
            if len(v):
                ship_lines.append(f"    {c}: {len(v)} ships, median of ship medians {v.median():.2f}")
        for a, b in CONTRASTS:
            if min(len(groups[a]), len(groups[b])) >= 5:
                dl, p = cliffs_delta(groups[a], groups[b])
                ship_lines.append(f"    {a} ships vs {b} ships: Cliff's d={dl:+.2f}, p={p:.3f} "
                                  f"(n={len(groups[a])}, {len(groups[b])} ships)")

    fig.suptitle("Ships: how concentrated, whether the gap survives without mega-ships, and ship-by-ship view",
                 y=1.0, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "11_ship_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '11_ship_analysis.png'}")

    lines = [
        "WHAT IT SHOWS: ship = the FIRST romantic ('/') relationship listed on a work (names order-insensitive).",
        "Row 1: how much of each category is a handful of ships. Row 2: the M/M advantage in bookmarks per hit",
        "after removing each group's biggest 1, 3, 5, 10 ships (the groups share no works): if the line stays above",
        "zero, the gap is not driven by a few mega-ships. Row 3: every ship as one point (size vs median bookmark",
        "rate), so M/M and F/F ships can be compared at the same popularity"
        + ("; marker shape = your canon label." if has_canon else ". Fill in " + CANON_FILE + " to add canon status."),
        "HOW CONCENTRATED, AND WHICH SHIPS DOMINATE (use these names to fill in the canon file):",
    ] + conc_lines + ["LEAVE-THE-BIGGEST-SHIPS-OUT (Cliff's delta; positive = first group higher):"] + leave_lines + \
        ["SHIP-LEVEL COMPARISON (every ship counts once, however many works it has):"] + ship_lines + [
        "CAUTION: only works that list a romantic ship are included, and only the first one is used, so works",
        "about several ships are filed under one. Ship popularity here is your sample's count, not AO3's.",
        "Ship-level groups are small, so treat those p-values as rough guides."]
    summary.add(title, lines)

# ------------------------------------------------------------
# Figure 12: category gaps before and after controls (no ship data needed)
# PASTE this block into analyser_grapher.py, right after plot_ship_analyses().
# Then in main(), right after the run_models(...) line, add:
#     plot_no_ship_summary(uw0, OUTPUT_DIR, summary)
# Untested on real data. Uses helpers already in the grapher:
# _model_frame, _fit_glm, _contrast, _drop_redundant, CATEGORY_NAMES, CONTRASTS.
# ------------------------------------------------------------

def _fit_unadjusted(d, outcome, pooled):
    """Category terms only (plus fandom when pooled): the 'before controls' version."""
    X = d[[f"cat_{c}" for c in CATEGORY_NAMES]].copy()
    if pooled:
        fd = pd.get_dummies(d["fandom_display"], prefix="fandom").astype(float)
        X = pd.concat([X, fd.iloc[:, 1:]], axis=1)
    X = X.loc[:, X.std() > 0]
    X = _drop_redundant(sm.add_constant(X, has_constant="add"))
    hits = d["hits"].to_numpy(float)
    y = np.minimum(d[outcome].to_numpy(float), hits) / hits
    model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=hits)
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        return model.fit(cov_type="HC1")


def plot_no_ship_summary(uw, output_dir, summary):
    title = "12_no_ship_summary.png - Category gaps before and after controls (no ship data needed)"
    if sm is None:
        print("!!! Figure 12 skipped: statsmodels missing (pip install statsmodels)")
        summary.add(title, ["SKIPPED: statsmodels is not installed."])
        return

    d = _model_frame(uw)
    scopes = ["All fandoms"] + list(d["fandom_display"].unique())
    outcomes = [("bookmarks", "Bookmarks per visit"), ("kudos", "Kudos per visit")]

    results = []
    for outcome, _ in outcomes:
        for scope in scopes:
            sub = d if scope == "All fandoms" else d[d["fandom_display"] == scope]
            if len(sub) < 200:
                continue
            pooled = scope == "All fandoms"
            fits = {
                "Before controls": _fit_unadjusted(sub, outcome, pooled),
                "After controls": _fit_glm(sub, outcome, pooled, False, False),
            }
            for kind, res in fits.items():
                for a, b in CONTRASTS:
                    ct = _contrast(res, a, b)
                    if ct:
                        results.append({"outcome": outcome, "scope": scope, "kind": kind,
                                        "a": a, "b": b, "n": int(res.nobs), **ct})

    if not results:
        print("!!! Figure 12 skipped: not enough works to fit the models")
        summary.add(title, ["SKIPPED: not enough works to fit the models."])
        return

    rows = [(s, a, b) for s in scopes for a, b in CONTRASTS]
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(18, 0.5 * len(rows) + 3), sharey=True)
    for ax, (outcome, ttl) in zip(axes, outcomes):
        for r in results:
            if r["outcome"] != outcome:
                continue
            before = r["kind"] == "Before controls"
            color = CONTRAST_COLORS[f"{r['a']} vs {r['b']}"]
            y = ypos[(r["scope"], r["a"], r["b"])] + (0.14 if before else -0.14)
            ax.errorbar(r["rr"], y, xerr=[[r["rr"] - r["lo"]], [r["hi"] - r["rr"]]],
                        fmt="o" if before else "s", color=color,
                        mfc="white" if before else color, capsize=2, markersize=6)
        ax.axvline(1, color="black", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(f"{ttl}\n(right of 1 = first group converts more visits)", fontsize=12)
        ax.set_xlabel("Rate ratio (log scale)")
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([f"{s}: {a} vs {b}" for s, a, b in rows], fontsize=10)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=3, label=k) for k, c in CONTRAST_COLORS.items()] + [
        Line2D([0], [0], marker="o", color="gray", mfc="white", lw=0, label="Before controls"),
        Line2D([0], [0], marker="s", color="gray", lw=0,
               label="After length, chapters, hits, rating, warnings")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.suptitle("Do the category gaps survive controls? (no relationship/ship data used)", y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "12_no_ship_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '12_no_ship_summary.png'}")

    lines = [
        "WHAT IT SHOWS: the same category gaps as figure 10, but only two versions: before any controls",
        "(hollow circles) and after controlling for length, chapters, hits, rating, warnings and fandom",
        "(filled squares). If a hollow and filled point sit close together, those controls did not explain",
        "the gap. No ship or canon information is used.",
        "NUMBERS (rate ratio, 95% interval; 1.00 = no difference):",
    ]
    for outcome, ttl in outcomes:
        lines.append(f"{ttl.upper()}:")
        for scope in scopes:
            for a, b in CONTRASTS:
                pair = {r["kind"]: r for r in results
                        if r["outcome"] == outcome and r["scope"] == scope and r["a"] == a and r["b"] == b}
                if len(pair) == 2:
                    u, c = pair["Before controls"], pair["After controls"]
                    lines.append(f"  [{scope}] {a} vs {b}: before x{u['rr']:.2f} ({u['lo']:.2f}-{u['hi']:.2f}); "
                                 f"after x{c['rr']:.2f} ({c['lo']:.2f}-{c['hi']:.2f})")
    lines.append("CAUTION: association, not cause. Same-author works are not independent, so intervals are")
    lines.append("somewhat too narrow. Ship popularity and canon status are NOT controlled here.")
    summary.add(title, lines)

# ============================================================
# Main
# ============================================================

def main():
    if not Path(CSV_PATH).exists():
        print(f"Could not find {CSV_PATH}. Pass the path as an argument:")
        print("  python3 analyser_grapher.py /path/to/ao3_results.csv")
        return

    notes = []
    df = load_data(CSV_PATH, notes)

    if not {"F/F", "M/M"} <= set(CATEGORY_ORDER):
        print("Need at least some F/F and some M/M cohort rows. Nothing to plot.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("\n--> Mode: RAW DATA (Unfiltered, Uncapped)")
    print(f"--> Groups drawn: {CATEGORY_ORDER}")

    check_filter_consistency(df, notes)
    summarize_sample(df, notes)

    # Unweighted unique works (plus ship columns) for the models and ship analyses below.
    uw0 = add_ship_columns(unique_works(df))

    # --- Real AO3 supply: reweight the 'All works' group to the true category mix ---
    supply, table, meta = None, [], {}
    if collect_supply is None:
        notes.append(f"WARNING: supply module unavailable ({SUPPLY_IMPORT_ERROR}); 'All works' is unweighted.")
    else:
        supply = collect_supply(
            unique_works(df), refresh=ARGS.refresh_supply, no_fetch=ARGS.no_fetch,
            scraper_path=ARGS.scraper_path, notes=notes,
        )
    if supply:
        df, table, meta, weight_notes = apply_supply_weights(df, supply, ALL_LABEL)
        notes += weight_notes
        if meta:
            WEIGHTED_ALL["on"] = True
            notes.append("'All works' is REWEIGHTED to AO3's real category mix (weighted resample, seed 2026).")
    if not WEIGHTED_ALL["on"]:
        notes.append("WARNING: 'All works' is the UNWEIGHTED sample (mix set by how many pages you scraped).")

    summary = Summary()
    if meta:
        plot_sample_vs_supply(table, meta, supply, OUTPUT_DIR, summary)
    plot_ecdfs(df, OUTPUT_DIR, summary)
    plot_distributions(df, OUTPUT_DIR, summary)
    plot_word_normalized(df, OUTPUT_DIR, summary)
    plot_rating_proxy(df, OUTPUT_DIR, summary)
    plot_warnings_proxy(df, OUTPUT_DIR, summary)
    plot_length_relationship(df, OUTPUT_DIR, summary)
    plot_supply_discovery_demand(df, OUTPUT_DIR, summary)
    run_group_comparisons(df, summary)
    plot_no_ship_summary(uw0, OUTPUT_DIR, summary)
    
    # --- deeper analyses: single- vs multi-category, joint models, ships / canon ---
    safe("09 single vs multi", plot_pure_vs_mixed, df, OUTPUT_DIR, summary)
    ships = ship_table(uw0)
    print(f"Ships found: {len(ships)} rows; relationships non-empty: "
        f"{uw0['relationships'].notna().sum() if 'relationships' in uw0 else 'NO COLUMN'}")
    canon = load_canon_labels()
    if canon is not None:
        uw0 = attach_canon(uw0, canon)
        notes.append(f"Canon labels loaded for {len(canon)} ships from {CANON_FILE}.")
    safe("10 joint model", run_models, uw0, canon is not None, OUTPUT_DIR, summary)
    safe("11 ship analysis", plot_ship_analyses, uw0, ships, canon is not None, OUTPUT_DIR, summary)
    if canon is None:
        safe("canon template", write_canon_template, ships)

    header = (
        [f"Data file: {CSV_PATH}", f"Figures folder: {OUTPUT_DIR}/",
         f"Groups: {', '.join(CATEGORY_ORDER)} ('{ALL_LABEL}' = every unique cohort work, any category; "
         + ("reweighted to AO3's real category mix)" if WEIGHTED_ALL["on"] else "NOT reweighted)"),
         "DATA NOTES:"]
        + ["  " + n for n in notes]
    )
    summary.write(SUMMARY_FILE, header)

    print(f"\nAll figures successfully saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()