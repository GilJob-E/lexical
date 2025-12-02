"""Generate visualization charts for feature statistics.

Creates 6 PNG charts for documentation:
1. feature_means.png - Horizontal bar chart with error bars
2. feature_distributions.png - Box plot with percentiles
3. zscore_interpretation.png - Z-Score interpretation guide
4. speaking_rate_dist.png - Speaking rate distribution curves
5. lexical_ratios_dist.png - Lexical ratio distribution curves
6. korean_findings.png - Korean language findings comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"
STATS_FILE = Path(__file__).parent.parent / "ko_liwc" / "data" / "feature_statistics.json"
DPI = 150

# Feature categories
SPEAKING_RATE_FEATURES = ["wpsec", "upsec", "fpsec"]
LEXICAL_FEATURES = ["quantifier_ratio", "we_ratio", "work_ratio", "adverb_ratio", "preposition_ratio"]

# Colors
COLORS = {
    "speaking_rate": "#3498db",  # Blue
    "lexical": "#2ecc71",        # Green
    "highlight": "#e74c3c",      # Red
    "neutral": "#95a5a6",        # Gray
}

# Feature display names
FEATURE_NAMES = {
    "wpsec": "Words/sec",
    "upsec": "Unique words/sec",
    "fpsec": "Fillers/sec",
    "quantifier_ratio": "Quantifier ratio",
    "we_ratio": "We ratio",
    "work_ratio": "Work ratio",
    "adverb_ratio": "Adverb ratio",
    "preposition_ratio": "Preposition ratio",
}


def load_statistics() -> Dict[str, Any]:
    """Load statistics from JSON file."""
    with open(STATS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_style():
    """Set up matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_feature_means(stats: Dict[str, Any]) -> None:
    """Chart 1: Horizontal bar chart with error bars (mean ± std)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    statistics = stats["statistics"]
    features = list(FEATURE_NAMES.keys())

    means = [statistics[f]["mean"] for f in features]
    stds = [statistics[f]["std"] for f in features]
    labels = [FEATURE_NAMES[f] for f in features]

    # Colors by category
    colors = [COLORS["speaking_rate"] if f in SPEAKING_RATE_FEATURES else COLORS["lexical"]
              for f in features]

    y_pos = np.arange(len(features))

    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8,
            error_kw={'ecolor': '#333', 'capsize': 3, 'capthick': 1})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value (mean ± std)')
    ax.set_title('Feature Statistics: Mean ± Standard Deviation\n(n=76,100 Korean interview samples)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["speaking_rate"], label='Speaking Rate'),
        Patch(facecolor=COLORS["lexical"], label='Lexical Ratio')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_means.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created feature_means.png")


def plot_feature_distributions(stats: Dict[str, Any]) -> None:
    """Chart 2: Box plot with percentiles (P5, P25, P50, P75, P95)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    statistics = stats["statistics"]
    features = list(FEATURE_NAMES.keys())

    # Create custom box plot data from percentiles
    positions = np.arange(len(features))

    for i, feat in enumerate(features):
        s = statistics[feat]
        p = s["percentiles"]

        # Box: P25 to P75, whiskers: P5 to P95
        box_props = dict(facecolor=COLORS["speaking_rate"] if feat in SPEAKING_RATE_FEATURES else COLORS["lexical"],
                        alpha=0.6)

        # Draw custom box
        box_bottom = p["p25"]
        box_top = p["p75"]
        whisker_bottom = p["p5"]
        whisker_top = p["p95"]
        median = p["p50"]
        mean = s["mean"]

        # Box
        ax.fill_between([i-0.3, i+0.3], box_bottom, box_top,
                       color=COLORS["speaking_rate"] if feat in SPEAKING_RATE_FEATURES else COLORS["lexical"],
                       alpha=0.6)
        # Median line
        ax.hlines(median, i-0.3, i+0.3, colors='black', linewidth=2)
        # Mean marker
        ax.scatter(i, mean, marker='D', color=COLORS["highlight"], s=40, zorder=5, label='Mean' if i == 0 else '')
        # Whiskers
        ax.vlines(i, whisker_bottom, box_bottom, colors='gray', linewidth=1)
        ax.vlines(i, box_top, whisker_top, colors='gray', linewidth=1)
        # Whisker caps
        ax.hlines(whisker_bottom, i-0.1, i+0.1, colors='gray', linewidth=1)
        ax.hlines(whisker_top, i-0.1, i+0.1, colors='gray', linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([FEATURE_NAMES[f] for f in features], rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Feature Distributions: Percentiles (P5, P25, P50, P75, P95)\n(n=76,100 samples)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["speaking_rate"], alpha=0.6, label='Speaking Rate (P25-P75)'),
        Patch(facecolor=COLORS["lexical"], alpha=0.6, label='Lexical Ratio (P25-P75)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS["highlight"],
                   markersize=8, label='Mean'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Median (P50)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created feature_distributions.png")


def plot_zscore_interpretation(stats: Dict[str, Any]) -> None:
    """Chart 3: Z-Score interpretation guide."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Z-score zones
    zones = [
        (-3, -2, '#e74c3c', 'Very Low\n(P2.3-)'),
        (-2, -1, '#f39c12', 'Low\n(P2.3-P16)'),
        (-1, 1, '#2ecc71', 'Average\n(P16-P84)'),
        (1, 2, '#f39c12', 'High\n(P84-P97.7)'),
        (2, 3, '#e74c3c', 'Very High\n(P97.7+)'),
    ]

    for z_min, z_max, color, label in zones:
        ax.axvspan(z_min, z_max, alpha=0.3, color=color)
        ax.text((z_min + z_max) / 2, 0.5, label, ha='center', va='center',
               fontsize=10, fontweight='bold')

    # Reference lines
    for z in [-2, -1, 0, 1, 2]:
        ax.axvline(z, color='gray', linestyle='--', alpha=0.7)
        ax.text(z, 1.05, f'z={z}', ha='center', fontsize=9)

    # Example values for wpsec
    wpsec_mean = stats["statistics"]["wpsec"]["mean"]
    wpsec_std = stats["statistics"]["wpsec"]["std"]

    example_values = [
        (-2, wpsec_mean - 2*wpsec_std, 'wpsec'),
        (-1, wpsec_mean - 1*wpsec_std, ''),
        (0, wpsec_mean, ''),
        (1, wpsec_mean + 1*wpsec_std, ''),
        (2, wpsec_mean + 2*wpsec_std, ''),
    ]

    ax.plot([-2, -1, 0, 1, 2], [wpsec_mean - 2*wpsec_std, wpsec_mean - wpsec_std,
                                 wpsec_mean, wpsec_mean + wpsec_std, wpsec_mean + 2*wpsec_std],
            'ko-', markersize=8, label=f'wpsec example (μ={wpsec_mean:.2f}, σ={wpsec_std:.2f})')

    # Add value annotations
    for z, val, _ in example_values:
        ax.annotate(f'{val:.2f}', (z, val), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Normalized scale')
    ax.set_title('Z-Score Interpretation Guide\nFormula: z = (x - μ) / σ')
    ax.legend(loc='upper left')

    # Hide y-axis ticks
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zscore_interpretation.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created zscore_interpretation.png")


def plot_speaking_rate_dist(stats: Dict[str, Any]) -> None:
    """Chart 4: Speaking rate features distribution curves."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    statistics = stats["statistics"]

    for ax, feat in zip(axes, SPEAKING_RATE_FEATURES):
        s = statistics[feat]
        mean, std = s["mean"], s["std"]
        p = s["percentiles"]

        # Generate normal distribution curve
        x = np.linspace(mean - 4*std, mean + 4*std, 200)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        ax.fill_between(x, y, alpha=0.3, color=COLORS["speaking_rate"])
        ax.plot(x, y, color=COLORS["speaking_rate"], linewidth=2)

        # Percentile lines
        for pval, pname in [(p["p5"], "P5"), (p["p50"], "P50"), (p["p95"], "P95")]:
            ax.axvline(pval, color='gray', linestyle='--', alpha=0.7)
            ax.text(pval, ax.get_ylim()[1] * 0.95, pname, ha='center', fontsize=8)

        # Mean line
        ax.axvline(mean, color=COLORS["highlight"], linewidth=2)
        ax.text(mean, ax.get_ylim()[1] * 0.85, f'μ={mean:.3f}', ha='center', fontsize=9,
               color=COLORS["highlight"])

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{FEATURE_NAMES[feat]}\n(μ={mean:.3f}, σ={std:.3f})')

    fig.suptitle('Speaking Rate Features Distribution (Normal approximation)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "speaking_rate_dist.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created speaking_rate_dist.png")


def plot_lexical_ratios_dist(stats: Dict[str, Any]) -> None:
    """Chart 5: Lexical ratio features distribution curves."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    statistics = stats["statistics"]

    for i, feat in enumerate(LEXICAL_FEATURES):
        ax = axes[i]
        s = statistics[feat]
        mean, std = s["mean"], s["std"]
        p = s["percentiles"]

        # Generate normal distribution curve
        x = np.linspace(max(0, mean - 4*std), mean + 4*std, 200)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        ax.fill_between(x, y, alpha=0.3, color=COLORS["lexical"])
        ax.plot(x, y, color=COLORS["lexical"], linewidth=2)

        # Percentile lines
        for pval, pname in [(p["p5"], "P5"), (p["p50"], "P50"), (p["p95"], "P95")]:
            ax.axvline(pval, color='gray', linestyle='--', alpha=0.7)

        # Mean line
        ax.axvline(mean, color=COLORS["highlight"], linewidth=2)

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{FEATURE_NAMES[feat]}\n(μ={mean:.4f}, σ={std:.4f})')

    # Hide the 6th subplot
    axes[5].axis('off')

    fig.suptitle('Lexical Ratio Features Distribution (Normal approximation)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lexical_ratios_dist.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created lexical_ratios_dist.png")


def plot_korean_findings(stats: Dict[str, Any]) -> None:
    """Chart 6: Korean language characteristics findings."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    statistics = stats["statistics"]

    # Left: we_ratio distribution (special case - mostly zeros)
    ax1 = axes[0]
    we_stats = statistics["we_ratio"]

    # Bar chart showing percentiles
    percentile_labels = ['P5', 'P25', 'P50', 'P75', 'P95', 'Mean']
    percentile_values = [
        we_stats["percentiles"]["p5"],
        we_stats["percentiles"]["p25"],
        we_stats["percentiles"]["p50"],
        we_stats["percentiles"]["p75"],
        we_stats["percentiles"]["p95"],
        we_stats["mean"]
    ]

    colors = [COLORS["neutral"]] * 5 + [COLORS["highlight"]]
    bars = ax1.bar(percentile_labels, percentile_values, color=colors, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, percentile_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Ratio')
    ax1.set_title('we_ratio: Korean Interview Characteristic\n(P50=0, Most samples have no "we" usage)')
    ax1.set_ylim(0, max(percentile_values) * 1.3)

    # Annotation
    ax1.annotate('75% of samples\nhave we_ratio = 0',
                xy=(2, 0), xytext=(3.5, 0.004),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

    # Right: Speaking rate comparison
    ax2 = axes[1]

    features = ['wpsec', 'upsec', 'fpsec']
    means = [statistics[f]["mean"] for f in features]
    stds = [statistics[f]["std"] for f in features]

    x_pos = np.arange(len(features))
    bars = ax2.bar(x_pos, means, yerr=stds, color=COLORS["speaking_rate"],
                   alpha=0.8, capsize=5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([FEATURE_NAMES[f] for f in features])
    ax2.set_ylabel('Per second')
    ax2.set_title('Speaking Rate Features\n(Korean agglutinative language characteristics)')

    # Add reference annotation
    ax2.annotate('Korean speakers average\n~2.86 words/second',
                xy=(0, means[0]), xytext=(1.5, means[0] + 1),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

    fig.suptitle('Korean Language Characteristics in Interview Data\n(n=76,100 samples)',
                fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "korean_findings.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Created korean_findings.png")


def main():
    """Generate all charts."""
    print(f"Loading statistics from {STATS_FILE}...")
    stats = load_statistics()

    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("\nGenerating charts...")
    plot_feature_means(stats)
    plot_feature_distributions(stats)
    plot_zscore_interpretation(stats)
    plot_speaking_rate_dist(stats)
    plot_lexical_ratios_dist(stats)
    plot_korean_findings(stats)

    print(f"\n✅ All 6 charts generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
