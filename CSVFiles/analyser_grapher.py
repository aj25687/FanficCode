"""
Exploratory analysis of ao3_results.csv: M/M vs F/F engagement patterns.

IMPORTANT SCOPE NOTE:
This script works ONLY with metadata already in your CSV (hits, kudos,
ratings, warnings, word count, etc.). It does NOT have your eventual
manual relationship-dynamics coding (physical affection, vulnerability,
etc.) -- that data doesn't exist yet. What this script CAN do is use
crude but real proxies (rating, warnings, length) to see where a
closer look with manual coding is most likely to pay off, and to
establish the baseline engagement gap you're trying to explain.

Treat every plot here as HYPOTHESIS-GENERATING, not confirmatory.
See the "Statistical practices used, and why" section printed at the
end for a full rundown of the choices made and why.

Usage:
    python3 analyze_engagement.py [path-to-ao3_results.csv]
    (defaults to ao3_results.csv in the current directory)
"""

import sys
import warnings as pywarnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

pywarnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "ao3_results.csv"
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# Colorblind-safe, non-valenced pair (ColorBrewer Dark2 teal/orange)
# rather than red/blue -- red commonly reads as "warning/deficit" and
# blue as "calm/neutral" in data-viz convention, which isn't a neutral
# choice on a topic this charged. F/F listed first throughout (plain
# alphabetical order, not "M/M as the default category") so it isn't
# always the rightmost/second-class bar in every chart.
CATEGORY_PALETTE = {"F/F": "#D95F02", "M/M": "#1B9E77"}
CATEGORY_ORDER = ["F/F", "M/M"]

# Outliers are shown (small, pale) rather than hidden. Hiding them
# entirely (seaborn's showfliers=False default choice) would make it
# impossible to see if one category has more extreme breakout-hit
# works than the other -- which could itself be a meaningful part of
# the answer. Kept visually subdued so the box/IQR stays the primary
# readable signal.
FLIER_PROPS = dict(marker="o", markersize=3, alpha=0.35, markeredgewidth=0)

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


# ============================================================
# Load and clean
# ============================================================

def load_data(path):
    df = pd.read_csv(
        path,
        dtype={
            "work_id": str, "category_filter": str, "warning_filter": str,
            "crossover_filter": str, "chapter_filter": str,
            "completion_filter": str, "timeframe_filter": str,
        },
    )

    df = df[df["role"] == "cohort"].copy()

    # Dedup by work_id. The same work can legitimately appear multiple
    # times across different sampling sessions/seeds in your pipeline
    # (this is intentional in your scraper's design) -- but for honest
    # analysis, each physical work should be counted once. This
    # mirrors the drop_duplicates(subset="work_id") logic already used
    # in your own compute_and_save_distribution()/get_accumulated_cohort().
    df = df.drop_duplicates(subset=["work_id"], keep="first")

    # Use the ACTUAL reported category text, not category_filter.
    # category_filter only records what search filter you used to find
    # a work -- if you ever searched with "All categories", that
    # filter code doesn't tell you what a given work's real category
    # is. The `category` column is AO3's own ground-truth label for
    # the work, which is what you actually want to group by.
    df["category_clean"] = df["category"].fillna("").str.strip()

    # Keep works tagged with EXACTLY one of M/M or F/F. Works tagged
    # with multiple categories (e.g. "F/F, Gen" on a fic with a
    # secondary non-romantic relationship tag) are excluded from the
    # core comparison rather than guessed at -- mixing them in would
    # blur the two groups you're trying to compare cleanly.
    df = df[df["category_clean"].isin(["M/M", "F/F"])].copy()

    return df


def summarize_sample(df):
    print("=" * 60)
    print("SAMPLE SUMMARY (after dedup + clean M/M-or-F/F-only filter)")
    print("=" * 60)
    counts = df.groupby(["fandom", "category_clean"]).size().unstack(fill_value=0)
    print(counts)
    print()
    for fandom in df["fandom"].unique():
        n = (df["fandom"] == fandom).sum()
        if n < 20:
            print(
                f"  NOTE: '{fandom}' has only {n} works after cleaning. "
                f"Treat any comparison within this fandom as very rough "
                f"-- small-n groups are noisy."
            )
    print()


# ============================================================
# Plot 1: ECDFs -- the baseline engagement gap
# ============================================================

def plot_ecdfs(df):
    """
    Empirical CDFs rather than bar charts of means. A bar chart of the
    mean kudos-to-hits ratio hides the actual shape of the
    distribution -- engagement ratios are typically right-skewed (a
    few very popular works, a long tail of less popular ones), so two
    groups can have similar means but very different shapes. An ECDF
    shows the FULL distribution at once and directly answers "what
    fraction of M/M works beat X% engagement" vs the same for F/F --
    which is exactly the percentile framing your own tool already
    uses elsewhere in this project.
    """
    fandoms = df["fandom"].unique()
    fig, axes = plt.subplots(len(fandoms), len(ENGAGEMENT_METRICS), figsize=(22, 6 * len(fandoms)), squeeze=False)

    # Compute one shared x-axis max PER METRIC (column), across all
    # fandoms, applied to every row for that column. Without this,
    # each fandom row auto-scales independently, and the visual
    # WIDTH of the M/M-vs-F/F gap becomes incomparable across fandoms
    # even for the exact same metric -- a gap that looks dramatic in
    # one row and modest in another might just be an axis-scaling
    # artifact, not a real difference in how big the gap is.
    column_max = {col: df[col].max() for col, _ in ENGAGEMENT_METRICS}

    for i, fandom in enumerate(fandoms):
        fdf = df[df["fandom"] == fandom]
        for j, (col, label) in enumerate(ENGAGEMENT_METRICS):
            ax = axes[i][j]
            for cat in CATEGORY_ORDER:
                color = CATEGORY_PALETTE[cat]
                values = fdf.loc[fdf["category_clean"] == cat, col].dropna().sort_values()
                if len(values) == 0:
                    continue
                y = np.arange(1, len(values) + 1) / len(values)
                ax.plot(values, y, label=f"{cat} (n={len(values)})", color=color, linewidth=2)
            ax.set_title(f"{fandom}\n{label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Cumulative proportion")
            ax.legend(fontsize=9)
            ax.set_xlim(0, column_max[col] * 1.02)
            # Log-scale x-axis: engagement ratios are typically
            # right-skewed (long tail of highly-engaged outliers), so
            # a linear axis compresses most of the data into a sliver
            # near zero. Log-scale spreads the bulk of the data out
            # for readability. Guard against all-zero columns.
            #
            # Worth knowing: log-scaling doesn't just "zoom in" evenly
            # -- it visually STRETCHES differences in the low/typical
            # range and visually COMPRESSES differences out in the
            # high/viral-outlier range. If the real M/M-vs-F/F gap is
            # concentrated in typical-case engagement, log-scale will
            # make it look bigger than a linear axis would; if the gap
            # is actually concentrated in rare breakout hits, log-scale
            # will make it look smaller. Sanity-check anything
            # surprising here against the linear-scale box plots too.
            if (fdf[col] > 0).any():
                ax.set_xscale("symlog")

    fig.suptitle("Engagement distributions: M/M vs F/F (ECDF)", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_engagement_ecdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/01_engagement_ecdfs.png")


# ============================================================
# Plot 2: Box/violin plots with group sizes shown
# ============================================================

def plot_distributions(df):
    """
    Box plots (median + IQR) rather than bar charts of means, for the
    same right-skew reason as above -- the median is a more honest
    "typical value" than the mean when outliers can drag the mean up.
    Faceted by fandom rather than pooled: pooling fandoms together
    risks Simpson's-paradox-style confounding, where a real pattern
    within each fandom gets hidden or reversed once mixed together
    (especially relevant here since your two fandoms differ a lot in
    size and M/M:F/F ratio).
    """
    fig, axes = plt.subplots(1, len(ENGAGEMENT_METRICS), figsize=(24, 7))
    for ax, (col, label) in zip(axes, ENGAGEMENT_METRICS):
        sns.boxplot(
            data=df, x="fandom", y=col, hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

        # Annotate group sizes directly on the plot. Never show a
        # summary comparison without the reader being able to see how
        # much data it's based on -- a dramatic-looking median
        # difference from 8 works means something very different than
        # the same difference from 300 works.
        for k, fandom in enumerate(df["fandom"].unique()):
            for cat, offset in [("M/M", -0.2), ("F/F", 0.2)]:
                n = ((df["fandom"] == fandom) & (df["category_clean"] == cat)).sum()
                ax.text(k + offset, ax.get_ylim()[1] * 0.95, f"n={n}", ha="center", fontsize=8, color="gray")

    fig.suptitle("Engagement ratios by category and fandom (box = median/IQR)", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_engagement_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/02_engagement_boxplots.png")


# ============================================================
# Plot 3: Word-normalized metrics (text works only)
# ============================================================

MIN_WORDS_FOR_NORMALIZATION = 500


def plot_word_normalized(df):
    """
    Restricted to is_text_work == True AND words >= MIN_WORDS_FOR_NORMALIZATION.

    is_text_work alone (words > 0) isn't a high enough bar: dividing
    by a SMALL word count massively amplifies the ratio -- a 150-word
    drabble with a handful of kudos can produce a per-10k-words value
    in the tens of thousands, purely from dividing by a tiny
    denominator, not from genuinely exceptional engagement. Once
    outliers are shown on the chart (rather than hidden, per the
    earlier fix), even one or two such points blow out the whole
    y-axis and make every other box invisible at the bottom. A
    word-count floor removes the mechanism that creates these
    artifacts in the first place, rather than just hiding their
    symptom.
    """
    text_df = df[(df["is_text_work"] == True) & (df["words"] >= MIN_WORDS_FOR_NORMALIZATION)].copy()  # noqa: E712
    n_excluded = ((df["is_text_work"] == True) & (df["words"] < MIN_WORDS_FOR_NORMALIZATION)).sum()  # noqa: E712

    fig, axes = plt.subplots(1, len(WORD_NORMALIZED_METRICS), figsize=(22, 7))
    for ax, (col, label) in zip(axes, WORD_NORMALIZED_METRICS):
        sns.boxplot(
            data=text_df, x="fandom", y=col, hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

        # Even with the word-count floor, a genuine extreme case can
        # still occur. As a second line of defense, clip the visible
        # y-range to the 1st-99th percentile of THIS metric's values
        # -- outlier points beyond that are still plotted (via
        # showfliers=True above) and will simply sit above the
        # visible area with a note, rather than being hidden or
        # allowed to wreck the axis for everything else.
        values = text_df[col].dropna()
        if len(values) > 10:
            lo, hi = np.percentile(values, [1, 99])
            if hi > lo:
                headroom = (hi - lo) * 0.15
                ax.set_ylim(max(0, lo - headroom), hi + headroom)
                n_above = (values > hi).sum()
                if n_above > 0:
                    ax.text(
                        0.5, 0.98, f"{n_above} point(s) above view range",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="dimgray",
                    )

    fig.suptitle("Word-normalized engagement (text works, \u2265" f"{MIN_WORDS_FOR_NORMALIZATION} words)", y=1.03, fontsize=16)
    if n_excluded > 0:
        fig.text(
            0.5, -0.03,
            f"{n_excluded} very short work(s) (<{MIN_WORDS_FOR_NORMALIZATION} words) excluded: "
            f"per-10k-words ratios are unstable for very short texts.",
            ha="center", fontsize=10, style="italic", color="dimgray",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_word_normalized_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/03_word_normalized_boxplots.png")


# ============================================================
# Plot 4: Rating as a crude physical-content proxy
# ============================================================

PROXY_CAVEAT = (
    "CRUDE PROXY -- not your actual coded variable. Hypothesis-generating only."
)


def plot_rating_proxy(df):
    """
    Tests a narrow slice of your core sub-question with data you
    already have: does being Explicit/Mature-rated (a crude stand-in
    for physical/sexual content, NOT the same as your planned
    "physical affection" coding -- rating is about explicitness
    level, not about affectionate behavior specifically) correlate
    with a BIGGER engagement boost for M/M than for F/F? If so,
    that's a thread worth pulling on with real coding later. If not,
    that's useful too -- it would suggest explicitness itself isn't
    the differentiator.

    Faceted by fandom (not pooled): pooling here would risk exactly
    the Simpson's-paradox-style confounding this script avoids
    everywhere else -- if the two fandoms have different rating
    distributions or different M/M:F/F baseline ratios, a pooled
    chart could show a pattern that doesn't hold within either
    fandom individually.
    """
    rating_order = ["General Audiences", "Teen And Up Audiences", "Mature", "Explicit", "Not Rated"]
    present_ratings = [r for r in rating_order if r in df["rating"].unique()]
    fandoms = df["fandom"].unique()

    fig, axes = plt.subplots(1, len(fandoms), figsize=(10 * len(fandoms), 7), squeeze=False)
    axes = axes[0]
    for ax, fandom in zip(axes, fandoms):
        fdf = df[(df["fandom"] == fandom) & (df["rating"].isin(present_ratings))]
        sns.boxplot(
            data=fdf, x="rating", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER,
            order=present_ratings, palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(fandom, fontsize=13)
        ax.set_ylabel("Kudos / Hits (%)")
        ax.set_xlabel("Rating")
        ax.tick_params(axis="x", rotation=15)

        # Splitting by rating AND category AND fandom can easily
        # produce small cells even when the fandom total looks fine
        # (e.g. 150 works but only 6 of them are Explicit-rated F/F).
        # Annotate every cell's n directly -- a dramatic-looking box
        # built on 4 works is noise, not a finding, and shouldn't be
        # presented with the same visual confidence as a box built on 60.
        ymax = ax.get_ylim()[1]
        for k, rating in enumerate(present_ratings):
            for cat, offset in [(CATEGORY_ORDER[0], -0.2), (CATEGORY_ORDER[1], 0.2)]:
                n = ((fdf["rating"] == rating) & (fdf["category_clean"] == cat)).sum()
                color = "red" if n < 10 else "gray"
                ax.text(k + offset, ymax * 0.97, f"n={n}", ha="center", fontsize=8, color=color)

    fig.suptitle("Kudos/Hits by rating and category, per fandom", y=1.05, fontsize=16)
    fig.text(
        0.5, -0.07,
        "Rating measures explicitness, not physical affection specifically. Explicit-rated works may\n"
        "also see different hit-counting behavior (e.g. more logged-out/private browsing), which can\n"
        "shift kudos-to-hits independently of actual content -- don't read this as a clean content effect.",
        ha="center", fontsize=10, style="italic", color="dimgray",
    )
    fig.text(0.5, -0.12, PROXY_CAVEAT, ha="center", fontsize=11, style="italic", color="dimgray")
    fig.text(0.02, 0.01, "n in red = fewer than 10 works in that cell; treat as noise.", fontsize=8, color="red")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_rating_proxy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/04_rating_proxy.png")


# ============================================================
# Plot 5: Warnings as a crude hurt/comfort-intensity proxy
# ============================================================

def plot_warnings_proxy(df):
    """
    Crude proxy for emotionally intense content (hurt/comfort-adjacent):
    does a work carry "Major Character Death" or "Graphic Depictions
    Of Violence" in its warnings field? Not the same as actually
    coding for hurt/comfort scenes, but cheap to check now.

    Faceted by fandom for the same Simpson's-paradox reason as the
    rating proxy above.
    """
    df = df.copy()
    df["has_intense_warning"] = df["warnings"].fillna("").str.contains(
        "Major Character Death|Graphic Depictions Of Violence", case=False
    )
    fandoms = df["fandom"].unique()

    fig, axes = plt.subplots(1, len(fandoms), figsize=(7 * len(fandoms), 7), squeeze=False)
    axes = axes[0]
    for ax, fandom in zip(axes, fandoms):
        fdf = df[df["fandom"] == fandom]
        sns.boxplot(
            data=fdf, x="has_intense_warning", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No intense\nwarning", "Death / Violence\npresent"])
        ax.set_title(fandom, fontsize=13)
        ax.set_ylabel("Kudos / Hits (%)")
        ax.set_xlabel("")

        ymax = ax.get_ylim()[1]
        for k, has_warning in enumerate([False, True]):
            for cat, offset in [(CATEGORY_ORDER[0], -0.2), (CATEGORY_ORDER[1], 0.2)]:
                n = ((fdf["has_intense_warning"] == has_warning) & (fdf["category_clean"] == cat)).sum()
                color = "red" if n < 10 else "gray"
                ax.text(k + offset, ymax * 0.97, f"n={n}", ha="center", fontsize=8, color=color)

    fig.suptitle("Kudos/Hits by intense-content warning and category, per fandom", y=1.05, fontsize=15)
    fig.text(
        0.5, -0.08,
        "Major Character Death / Graphic Violence indicate dark or tragic content, not specifically\n"
        "hurt/comfort or emotional vulnerability -- a warning for violence doesn't mean the fic contains\n"
        "healing, aftercare, or tender vulnerability beats; it could just as easily be an unresolved tragedy.",
        ha="center", fontsize=10, style="italic", color="dimgray",
    )
    fig.text(0.5, -0.13, PROXY_CAVEAT, ha="center", fontsize=11, style="italic", color="dimgray")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_warnings_proxy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/05_warnings_proxy.png")


# ============================================================
# Plot 6: Word count as a proxy for slow-burn/relationship development room
# ============================================================

def plot_length_relationship(df):
    """
    Scatter + binned-median trend line of engagement vs. word count,
    separately by category. Longer works have more room for the kind
    of relationship-development beats (vulnerability, hurt/comfort
    arcs) your eventual coding will capture -- this checks whether
    length itself already predicts engagement differently by
    category, which would be a confound to control for once you do
    the real coding.

    Uses a binned-median trend line rather than a smoothed regression
    curve (e.g. LOWESS): it needs no extra statistical dependencies,
    it's trivial to see exactly how it's computed, and the median
    within each bin is already robust to the outliers that are common
    in engagement data -- a reasonable, honest default when you don't
    need a publication-grade smoother.

    Bin edges are computed ONCE on the pooled (both-category) word
    count distribution, then applied identically to both categories --
    not computed separately per category. If M/M and F/F have
    meaningfully different word-count distributions (e.g. M/M skewing
    toward longer multi-chapter fic), separate per-category qcut calls
    would produce DIFFERENT bin boundaries for each line, so a given
    x-position wouldn't represent the same word-count window for both
    categories, making direct comparison at any given point misleading.
    Shared bins fix that: both lines are evaluated at the same
    word-count checkpoints.
    """
    text_df = df[(df["is_text_work"] == True) & (df["words"] > 0)].copy()  # noqa: E712
    text_df["log_words"] = np.log10(text_df["words"])

    fig, ax = plt.subplots(figsize=(12, 8))
    n_bins = 8

    shared_bin_edges = None
    if len(text_df) >= n_bins * 6:
        _, shared_bin_edges = pd.qcut(text_df["log_words"], q=n_bins, duplicates="drop", retbins=True)

    for cat in CATEGORY_ORDER:
        color = CATEGORY_PALETTE[cat]
        subset = text_df[text_df["category_clean"] == cat]
        ax.scatter(subset["log_words"], subset["kudos_to_hits"], alpha=0.25, s=20, color=color, label=f"{cat} (n={len(subset)})")

        if shared_bin_edges is not None and len(subset) >= n_bins:
            bins = pd.cut(subset["log_words"], bins=shared_bin_edges, duplicates="drop")
            binned = subset.groupby(bins, observed=True).agg(
                x=("log_words", "median"), y=("kudos_to_hits", "median"), n=("log_words", "size")
            ).sort_values("x")
            # Bins with very few points for this category are noisy --
            # still plotted (for full transparency) but visually
            # de-emphasized rather than given equal weight to well-
            # supported bins.
            for _, row in binned.iterrows():
                marker_size = 60 if row["n"] >= 5 else 25
                marker_alpha = 1.0 if row["n"] >= 5 else 0.4
                ax.scatter(row["x"], row["y"], color=color, s=marker_size, alpha=marker_alpha, zorder=5)
            ax.plot(binned["x"], binned["y"], color=color, linewidth=3, alpha=0.8)

    ax.set_xlabel("Word count (log10 scale)")
    ax.set_ylabel("Kudos / Hits (%)")
    ax.set_title("Engagement vs. word count, by category\n(line = median per word-count bin)")
    ax.legend()
    fig.text(
        0.5, -0.04,
        "Bins share the same word-count edges across both categories (so points are comparable at a given\n"
        "x-position), but are equal-FREQUENCY overall, not equal-width -- faint/small markers mark bins with\n"
        "fewer than 5 works for that category; treat those points as noisy, not a real local pattern.",
        ha="center", fontsize=10, style="italic", color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_length_vs_engagement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/06_length_vs_engagement.png")


# ============================================================
# Plot 7: Supply / discovery / demand decomposition
# ============================================================

def plot_supply_discovery_demand(df):
    """
    Splits "popularity" into three genuinely different things that
    are easy to conflate:
      Supply    -- how many works exist (fic count)
      Discovery -- how many people found them (hits)
      Demand    -- given discovery, did people like what they found
                   (kudos-to-hits ratio)
    A gap in supply doesn't necessarily mean a gap in demand -- if F/F
    works that exist have HIGH kudos-to-hits despite low counts, the
    story is "not enough gets written," not "readers don't want it."
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    counts = df.groupby(["fandom", "category_clean"]).size().reset_index(name="count")
    sns.barplot(data=counts, x="fandom", y="count", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[0])
    axes[0].set_title("Supply: work count")
    axes[0].set_ylabel("Number of works")
    axes[0].set_xlabel("")

    sns.boxplot(data=df, x="fandom", y="hits", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[1], showfliers=True, flierprops=FLIER_PROPS)
    axes[1].set_title("Discovery: hits")
    axes[1].set_ylabel("Hits")
    axes[1].set_xlabel("")
    axes[1].set_yscale("log")

    sns.boxplot(data=df, x="fandom", y="kudos_to_hits", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[2], showfliers=True, flierprops=FLIER_PROPS)
    axes[2].set_title("Demand given discovery: kudos/hits")
    axes[2].set_ylabel("Kudos / Hits (%)")
    axes[2].set_xlabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Three different bottlenecks, not one 'popularity'", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_supply_discovery_demand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/07_supply_discovery_demand.png")


# ============================================================
# Supplementary: Mann-Whitney U tests (descriptive, NOT confirmatory)
# ============================================================

MIN_GROUP_N_FOR_TEST = 15


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg false discovery rate correction, implemented
    directly (no statsmodels dependency, which isn't guaranteed to be
    installed). Running many Mann-Whitney tests and reporting raw
    p-values invites exactly the problem you flagged: highlighting
    one "significant" result out of dozens without correction
    overstates how surprising that result actually is. BH correction
    adjusts each p-value upward based on how many tests were run and
    how it ranks among them, giving a fairer sense of which results
    would still look notable after accounting for the multiple-testing
    problem -- less conservative than a flat Bonferroni correction,
    which is a reasonable choice for exploratory/hypothesis-generating
    analysis like this rather than a single confirmatory test.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity (standard BH step-up procedure)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def run_group_comparisons(df):
    """
    Mann-Whitney U rather than a t-test: engagement ratios are
    unlikely to be normally distributed (bounded at 0, often
    right-skewed), and Mann-Whitney doesn't assume normality -- it
    compares whether one group's values tend to rank higher than the
    other's, which is a safer default for this kind of data.

    Requires at least MIN_GROUP_N_FOR_TEST (15) works per group, not
    just a handful -- a test run on 5 works per side can produce a
    p-value that looks meaningful but is really just noise from a tiny
    sample. All raw p-values are collected and passed through a
    Benjamini-Hochberg FDR correction before being reported, rather
    than just printing a verbal warning about multiple comparisons.
    """
    print("=" * 60)
    print("MANN-WHITNEY U COMPARISONS (M/M vs F/F), BY FANDOM")
    print("DESCRIPTIVE / HYPOTHESIS-GENERATING ONLY")
    print("=" * 60)

    all_metrics = ENGAGEMENT_METRICS + WORD_NORMALIZED_METRICS
    results = []

    for fandom in df["fandom"].unique():
        fdf = df[df["fandom"] == fandom]
        for col, label in all_metrics:
            mm = fdf.loc[fdf["category_clean"] == "M/M", col].dropna()
            ff = fdf.loc[fdf["category_clean"] == "F/F", col].dropna()
            if len(mm) < MIN_GROUP_N_FOR_TEST or len(ff) < MIN_GROUP_N_FOR_TEST:
                results.append({
                    "fandom": fandom, "label": label, "skipped": True,
                    "n_mm": len(mm), "n_ff": len(ff),
                })
                continue
            _, p = stats.mannwhitneyu(mm, ff, alternative="two-sided")
            results.append({
                "fandom": fandom, "label": label, "skipped": False,
                "n_mm": len(mm), "n_ff": len(ff), "p": p,
                "median_mm": mm.median(), "median_ff": ff.median(),
            })

    tested = [r for r in results if not r["skipped"]]
    if tested:
        adjusted = benjamini_hochberg([r["p"] for r in tested])
        for r, p_adj in zip(tested, adjusted):
            r["p_adj"] = p_adj

    for fandom in df["fandom"].unique():
        print(f"\n--- {fandom} ---")
        for r in results:
            if r["fandom"] != fandom:
                continue
            if r["skipped"]:
                print(f"  {r['label']:30s}: skipped (n too small: M/M={r['n_mm']}, F/F={r['n_ff']}, need >={MIN_GROUP_N_FOR_TEST})")
                continue
            flag = " **" if r["p_adj"] < 0.05 else ""
            print(
                f"  {r['label']:30s}: M/M median={r['median_mm']:.2f}, F/F median={r['median_ff']:.2f}, "
                f"raw p={r['p']:.4f}, FDR-adjusted p={r['p_adj']:.4f}{flag} "
                f"(n_MM={r['n_mm']}, n_FF={r['n_ff']})"
            )

    n_sig_raw = sum(1 for r in tested if r["p"] < 0.05)
    n_sig_adj = sum(1 for r in tested if r["p_adj"] < 0.05)
    print(
        f"\n{len(tested)} test(s) run. {n_sig_raw} significant at raw p<0.05; "
        f"{n_sig_adj} remain significant after Benjamini-Hochberg FDR correction (marked **).\n"
        f"Use the FDR-adjusted column, not the raw p-value, when deciding what's worth discussing."
    )


# ============================================================
# Main
# ============================================================

def main():
    if not Path(CSV_PATH).exists():
        print(f"Could not find {CSV_PATH}. Pass the path as an argument:")
        print("  python3 analyze_engagement.py /path/to/ao3_results.csv")
        return

    df = load_data(CSV_PATH)

    if df.empty:
        print("No M/M or F/F cohort rows found after cleaning. Nothing to plot.")
        return

    summarize_sample(df)

    print(
        "REMINDER: the four engagement metrics often don't all point the same "
        "direction (e.g. kudos/hits and comments/kudos can favor different "
        "categories within the same fandom). Check all four panels in each "
        "figure, not just the first one -- the first metric isn't more "
        "important than the others, it's just listed first.\n"
    )

    plot_ecdfs(df)
    plot_distributions(df)
    plot_word_normalized(df)
    plot_rating_proxy(df)
    plot_warnings_proxy(df)
    plot_length_relationship(df)
    plot_supply_discovery_demand(df)

    run_group_comparisons(df)

    print("\nAll figures saved to ./figures/")


if __name__ == "__main__":
    main()