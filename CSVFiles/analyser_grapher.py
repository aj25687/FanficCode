"""
Exploratory analysis of ao3_results.csv: M/M vs F/F engagement patterns.

Usage:
    python3 analyze_engagement.py [path-to-ao3_results.csv]
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
NORMALIZATION_SEED = 2026

sns.set_theme(style="whitegrid", context="talk")

CATEGORY_PALETTE = {"F/F": "#D95F02", "M/M": "#1B9E77"}
CATEGORY_ORDER = ["F/F", "M/M"]

FLIER_PROPS = dict(marker="o", markersize=3, alpha=0.35, markeredgewidth=0)
HUE_OFFSETS = {CATEGORY_ORDER[0]: -0.2, CATEGORY_ORDER[1]: 0.2}

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


# ============================================================
# Helper & Formatting Utilities
# ============================================================

def annotate_n(ax, x, n):
    color = "red" if n < 10 else "dimgray"
    ax.text(
        x, -0.015, f"n={n}", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, color=color, clip_on=False,
    )


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
    values = pd.Series(values).dropna()
    if len(values) < 10:
        return
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return
    pad = (hi - lo) * 0.12
    ax.set_ylim(max(0, lo - pad) if lo >= 0 else lo - pad, hi + pad)
    n_above = int((values > hi + pad).sum())
    if note and n_above > 0:
        ax.text(
            0.99, 0.99, f"{n_above} point(s) above view",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray",
        )


# ============================================================
# Load and Normalization
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
    df = df.drop_duplicates(subset=["work_id"], keep="first")
    df["category_clean"] = df["category"].fillna("").str.strip()
    df = df[df["category_clean"].isin(["M/M", "F/F"])].copy()

    # Safely create fandom_display if missing
    if "fandom_display" not in df.columns:
        df["fandom_display"] = df["fandom"]

    df["fandom_display"] = df["fandom_display"].fillna("").str.strip().str.split().str[0]

    df["date_updated_parsed"] = pd.to_datetime(df["date_updated"], format="mixed", dayfirst=True, errors="coerce")
    return df


def normalize_dataset(df):
    """
    Applies dynamic time cutoff and equalizes work counts across fandoms.
    """
    df_filtered = df.dropna(subset=["date_updated_parsed"]).copy()
    if df_filtered.empty:
        print("No valid update dates found for normalization.")
        return df

    latest_per_fandom = df_filtered.groupby("fandom_display")["date_updated_parsed"].max()
    dynamic_cutoff = latest_per_fandom.min()
    limiting_fandom = latest_per_fandom.idxmin()

    print("\n--- APPLYING TIME & SIZE NORMALIZATION ---")
    print(f"Dynamic cutoff date chosen: {dynamic_cutoff.strftime('%Y-%m-%d')} (set by '{limiting_fandom}')")

    df_filtered = df_filtered[df_filtered["date_updated_parsed"] <= dynamic_cutoff].copy()

    fandom_counts = df_filtered.groupby("fandom_display").size()
    min_count = fandom_counts.min()
    smallest_fandom = fandom_counts.idxmin()

    print(f"Smallest fandom sample: '{smallest_fandom}' with {min_count} works.")
    print(f"Subsampling all fandoms down to {min_count} works...\n")

    rng = np.random.RandomState(NORMALIZATION_SEED)
    subsampled_dfs = []
    for fandom, group in df_filtered.groupby("fandom_display"):
        if len(group) > min_count:
            idx = rng.choice(group.index, size=min_count, replace=False)
            subsampled_dfs.append(group.loc[idx])
        else:
            subsampled_dfs.append(group)

    return pd.concat(subsampled_dfs)


def summarize_sample(df):
    print("=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)
    counts = df.groupby(["fandom_display", "category_clean"]).size().unstack(fill_value=0)
    print(counts)
    print()


# ============================================================
# Plot 1: ECDFs
# ============================================================

def plot_ecdfs(df, output_dir):
    fandoms = df["fandom_display"].unique()
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
                ax.plot(values, y, label=f"{cat} (n={len(values)})", color=color, linewidth=2)
            ax.set_title(f"{fandom}\n{label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Cumulative proportion")
            ax.legend(title="Category (n)", fontsize=9, title_fontsize=9, loc="lower right")
            ax.set_xlim(0, column_max[col] * 1.02)
            if (fdf[col] > 0).any():
                ax.set_xscale("symlog")

    fig.suptitle("Engagement distributions: M/M vs F/F (ECDF)", y=1.02, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_dir / "01_engagement_ecdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '01_engagement_ecdfs.png'}")


# ============================================================
# Plot 2: Box plots by fandom and metric
# ============================================================

def plot_distributions(df, output_dir):
    fig, axes = plt.subplots(1, len(ENGAGEMENT_METRICS), figsize=(24, 7))
    for ax, (col, label) in zip(axes, ENGAGEMENT_METRICS):
        sns.boxplot(
            data=df, x="fandom_display", y=col, hue="category_clean", hue_order=CATEGORY_ORDER,
            palette=CATEGORY_PALETTE, ax=ax, showfliers=True, flierprops=FLIER_PROPS,
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        set_robust_ylim(ax, df[col])
        ax.tick_params(axis="x", pad=18)

        for k, fandom in enumerate(df["fandom_display"].unique()):
            for cat, offset in HUE_OFFSETS.items():
                n = ((df["fandom_display"] == fandom) & (df["category_clean"] == cat)).sum()
                annotate_n(ax, k + offset, n)

    use_shared_legend(fig, axes)
    fig.suptitle("Engagement ratios by category and fandom (box = median/IQR)", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "02_engagement_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '02_engagement_boxplots.png'}")


# ============================================================
# Plot 3: Word-normalized metrics (text works only)
# ============================================================

MIN_WORDS_FOR_NORMALIZATION = 500


def plot_word_normalized(df, output_dir):
    text_df = df[(df["is_text_work"] == True) & (df["words"] >= MIN_WORDS_FOR_NORMALIZATION)].copy()  # noqa: E712
    n_excluded = ((df["is_text_work"] == True) & (df["words"] < MIN_WORDS_FOR_NORMALIZATION)).sum()  # noqa: E712

    fig, axes = plt.subplots(1, len(WORD_NORMALIZED_METRICS), figsize=(22, 7))
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
            n_above = (text_df[col] > WORD_Y_LIMITS[col][1]).sum()
            if n_above > 0:
                ax.text(
                    0.99, 0.95, f"{n_above} point(s) above view",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray"
                )
        else:
            set_robust_ylim(ax, text_df[col])

        ax.tick_params(axis="x", pad=18)
        for k, fandom in enumerate(text_df["fandom_display"].unique()):
            for cat, offset in HUE_OFFSETS.items():
                n = ((text_df["fandom_display"] == fandom) & (text_df["category_clean"] == cat)).sum()
                annotate_n(ax, k + offset, n)

    use_shared_legend(fig, axes)
    fig.suptitle("Word-normalized engagement (text works, \u2265" f"{MIN_WORDS_FOR_NORMALIZATION} words)", y=1.03, fontsize=16)
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


# ============================================================
# Plot 4: Rating proxy
# ============================================================

PROXY_CAVEAT = "CRUDE PROXY -- not your actual coded variable. Hypothesis-generating only."


def plot_rating_proxy(df, output_dir):
    rating_order = ["General Audiences", "Teen And Up Audiences", "Mature", "Explicit", "Not Rated"]
    present_ratings = [r for r in rating_order if r in df["rating"].unique()]
    fandoms = df["fandom_display"].unique()

    fig, axes = plt.subplots(
        1, len(fandoms), figsize=(11 * len(fandoms), 8), squeeze=False, sharey=True
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
        ax.tick_params(axis="x", pad=18)

        for k, rating in enumerate(present_ratings):
            for cat, offset in HUE_OFFSETS.items():
                n = ((fdf["rating"] == rating) & (fdf["category_clean"] == cat)).sum()
                annotate_n(ax, k + offset, n)

    set_robust_ylim(axes[0], df.loc[df["rating"].isin(present_ratings), "kudos_to_hits"])
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


# ============================================================
# Plot 5: Warnings proxy
# ============================================================

def plot_warnings_proxy(df, output_dir):
    df = df.copy()
    df["has_intense_warning"] = df["warnings"].fillna("").str.contains(
        "Major Character Death|Graphic Depictions Of Violence", case=False
    )
    fandoms = df["fandom_display"].unique()

    fig, axes = plt.subplots(
        1, len(fandoms), figsize=(8 * len(fandoms), 8), squeeze=False, sharey=True
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
        ax.tick_params(axis="x", pad=18)

        for k, has_warning in enumerate([False, True]):
            for cat, offset in HUE_OFFSETS.items():
                n = ((fdf["has_intense_warning"] == has_warning) & (fdf["category_clean"] == cat)).sum()
                annotate_n(ax, k + offset, n)

    set_robust_ylim(axes[0], df["kudos_to_hits"])
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


# ============================================================
# Plot 6: Length relationship
# ============================================================

def plot_length_relationship(df, output_dir):
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
            for _, row in binned.iterrows():
                marker_size = 60 if row["n"] >= 5 else 25
                marker_alpha = 1.0 if row["n"] >= 5 else 0.4
                ax.scatter(row["x"], row["y"], color=color, s=marker_size, alpha=marker_alpha, zorder=5)
            ax.plot(binned["x"], binned["y"], color=color, linewidth=3, alpha=0.8)

    ax.set_xlabel("Word count (log10 scale)")
    ax.set_ylabel("Kudos / Hits (%)")
    ax.set_title("Engagement vs. word count, by category\n(line = median per word-count bin)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_length_vs_engagement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '06_length_vs_engagement.png'}")


# ============================================================
# Plot 7: Supply / discovery / demand
# ============================================================

def plot_supply_discovery_demand(df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    counts = df.groupby(["fandom_display", "category_clean"]).size().reset_index(name="count")
    sns.barplot(data=counts, x="fandom_display", y="count", hue="category_clean", hue_order=CATEGORY_ORDER, palette=CATEGORY_PALETTE, ax=axes[0])
    axes[0].set_title("Supply: work count")
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

    fig.suptitle("Three different bottlenecks, not one 'popularity'", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "07_supply_discovery_demand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / '07_supply_discovery_demand.png'}")


# ============================================================
# Supplementary: Mann-Whitney U tests
# ============================================================

MIN_GROUP_N_FOR_TEST = 15


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


def run_group_comparisons(df):
    print("=" * 60)
    print("MANN-WHITNEY U COMPARISONS (M/M vs F/F), BY FANDOM")
    print("DESCRIPTIVE / HYPOTHESIS-GENERATING ONLY")
    print("=" * 60)

    all_metrics = ENGAGEMENT_METRICS + WORD_NORMALIZED_METRICS
    results = []

    for fandom in df["fandom_display"].unique():
        fdf = df[df["fandom_display"] == fandom]
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

    for fandom in df["fandom_display"].unique():
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


# ============================================================
# Main Execution Loop with Terminal Prompt
# ============================================================

def main():
    if not Path(CSV_PATH).exists():
        print(f"Could not find {CSV_PATH}. Pass the path as an argument:")
        print("  python3 analyze_engagement.py /path/to/ao3_results.csv")
        return

    raw_df = load_data(CSV_PATH)

    if raw_df.empty:
        print("No M/M or F/F cohort rows found after cleaning. Nothing to plot.")
        return

    user_choice = input("\nNormalize the data? (y/n): ").strip().lower()

    if user_choice == "y":
        target_df = normalize_dataset(raw_df)
        output_dir = Path("figures_normalized")
        print("\n--> Mode: NORMALIZED DATA")
    else:
        target_df = raw_df
        output_dir = Path("figures_raw")
        print("\n--> Mode: RAW DATA (Unfiltered, Uncapped)")

    output_dir.mkdir(exist_ok=True)
    summarize_sample(target_df)

    plot_ecdfs(target_df, output_dir)
    plot_distributions(target_df, output_dir)
    plot_word_normalized(target_df, output_dir)
    plot_rating_proxy(target_df, output_dir)
    plot_warnings_proxy(target_df, output_dir)
    plot_length_relationship(target_df, output_dir)
    plot_supply_discovery_demand(target_df, output_dir)

    run_group_comparisons(target_df)

    print(f"\nAll 7 figures successfully saved to ./{output_dir}/")


if __name__ == "__main__":
    main()