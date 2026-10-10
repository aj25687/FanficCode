"""
Exploratory analysis of ao3_results.csv: engagement patterns by relationship category.

Groups shown in every graph:
    F/F        works whose category tag is exactly F/F
    M/M        works whose category tag is exactly M/M
    F/M        works whose category tag is exactly F/M (AO3's name for "M/F")
    All works  every unique cohort work in the CSV, whatever its category
               (including Gen, Multi, Other and multi-category works). It is a
               reference line, NOT a population estimate: its mix is set by how
               many pages you scraped for each category.

Usage:
    python3 analyser_grapher.py [path-to-ao3_results.csv]

Outputs:
    figures_raw/*.png   the 7 figures
    summary.txt         appended on every run: a timestamped plain-language
                        summary of each figure, with numbers from your data
"""

import sys
import warnings as pywarnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

pywarnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "ao3_results.csv"
OUTPUT_DIR = Path("figures_raw")
SUMMARY_FILE = "summary.txt"

sns.set_theme(style="whitegrid", context="talk")

ALL_LABEL = "All works"
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
        return f"{cat}: no data", None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    flag = "  [n<10: noisy]" if len(v) < MIN_N_NOISY else ""
    return f"{cat}: median {med:.2f} (IQR {q1:.2f}-{q3:.2f}, n={len(v)}){flag}", med


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
    return [Patch(facecolor=CATEGORY_PALETTE[c], edgecolor="black", label=c) for c in CATEGORY_ORDER]


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
    df = df.drop_duplicates(subset=["work_id"], keep="first")
    n_dupes = n_before - len(df)
    summary_notes.append(f"Cohort rows: {n_before}; duplicate work_ids dropped: {n_dupes}.")

    df["category_clean"] = df["category"].fillna("").str.strip()

    # Rows that belong to no single pure category (Gen, Multi, Other, blank, or
    # tagged with several categories). They count in 'All works' only.
    other = df.loc[~df["category_clean"].isin(PURE_CATEGORIES), "category_clean"]
    summary_notes.append(
        f"{len(other)} unique works are in none of {PURE_CATEGORIES} (Gen, Multi, Other, blank or "
        f"multi-category tags); they appear only in '{ALL_LABEL}'."
    )
    if len(other):
        top = other.replace("", "(blank)").value_counts().head(8)
        summary_notes.append("  Most common of those: " + "; ".join(f"{k}: {v}" for k, v in top.items()))

    # Stack: each pure category once, plus every unique work as 'All works'.
    pure = df[df["category_clean"].isin(PURE_CATEGORIES)].copy()
    everything = df.copy()
    everything["category_clean"] = ALL_LABEL
    df = pd.concat([pure, everything], ignore_index=True)

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
                    values, y, label=f"{cat} (n={len(values)})", color=color,
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
        f"WHAT IT SHOWS: kudos, bookmarks and comments per 10,000 words, for text works of at least",
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
    df["has_intense_warning"] = df["warnings"].fillna("").str.contains(
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

    for cat in CATEGORY_ORDER:
        color = CATEGORY_PALETTE[cat]
        subset = text_df[text_df["category_clean"] == cat]
        is_all = cat == ALL_LABEL
        # 'All works' would just repaint every dot, so it gets only its median line.
        if not is_all:
            ax.scatter(subset["log_words"], subset["kudos_to_hits"], alpha=0.2, s=18, color=color,
                       label=f"{cat} (n={len(subset)})")
        else:
            ax.plot([], [], color=color, linestyle="--", linewidth=3, label=f"{cat} median line (n={len(subset)})")

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
            lines.append(
                f"  {cat}: n={len(s)}, median words {s['words'].median():,.0f}, "
                f"Spearman rho(length, kudos/hits)={rho:.2f} (p={p:.3g}) -> {direction}"
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
                va = fdf.loc[fdf["category_clean"] == a, col].dropna()
                vb = fdf.loc[fdf["category_clean"] == b, col].dropna()
                base = {"fandom": fandom, "pair": (a, b), "label": label,
                        "n_a": len(va), "n_b": len(vb)}
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
                       f"raw p={r['p']:.4f}, FDR p={r['p_adj']:.4f}{flag} (n_{a}={r['n_a']}, n_{b}={r['n_b']})")
            print(txt)
            lines.append(txt)
    lines.append("CAUTION: works by the same author are not independent, so p-values are optimistic. A small")
    lines.append("p with a negligible Cliff's d is a detectable but unimportant difference.")
    summary.add("Statistical tests (printed to terminal; not a figure)", lines)


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

    summary = Summary()
    plot_ecdfs(df, OUTPUT_DIR, summary)
    plot_distributions(df, OUTPUT_DIR, summary)
    plot_word_normalized(df, OUTPUT_DIR, summary)
    plot_rating_proxy(df, OUTPUT_DIR, summary)
    plot_warnings_proxy(df, OUTPUT_DIR, summary)
    plot_length_relationship(df, OUTPUT_DIR, summary)
    plot_supply_discovery_demand(df, OUTPUT_DIR, summary)
    run_group_comparisons(df, summary)

    header = (
        [f"Data file: {CSV_PATH}", f"Figures folder: {OUTPUT_DIR}/",
         f"Groups: {', '.join(CATEGORY_ORDER)} ('{ALL_LABEL}' = every unique cohort work, any category)",
         "DATA NOTES:"]
        + ["  " + n for n in notes]
    )
    summary.write(SUMMARY_FILE, header)

    print(f"\nAll 7 figures successfully saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()