"""
Plot distributions of generated layouts and inflow conditions.

This script visualizes:
- Number of turbines per layout
- Layout types distribution
- Turbine spacings
- Inflow wind speeds
- Ambient turbulence intensities
- Sample layouts
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metadata(data_dir: Path):
    """Load layouts and inflows metadata from npz files."""
    layouts_path = data_dir / "layouts_metadata.npz"
    inflows_path = data_dir / "inflows_metadata.npz"

    if not layouts_path.exists():
        raise FileNotFoundError(f"Layouts file not found: {layouts_path}")
    if not inflows_path.exists():
        raise FileNotFoundError(f"Inflows file not found: {inflows_path}")

    layouts_data = np.load(layouts_path, allow_pickle=True)
    inflows_raw = np.load(inflows_path, allow_pickle=True)

    # Extract inflows from per-layout format to flat arrays
    n_layouts = int(inflows_raw["n_layouts"])
    all_ws = []
    all_ti = []
    for i in range(n_layouts):
        inflow_arr = inflows_raw[f"inflows_{i}"]  # Shape: (n_inflows, 2)
        all_ws.extend(inflow_arr[:, 0])
        all_ti.extend(inflow_arr[:, 1])

    inflows_data = {
        "wind_speeds": np.array(all_ws),
        "turbulence_intensities": np.array(all_ti),
        "n_inflows_per_layout": inflows_raw["inflows_0"].shape[0],
    }

    return layouts_data, inflows_data


def plot_layout_distributions(layouts_data, output_dir: Path):
    """Plot distributions of layout properties."""
    # Extract metadata
    n_turbines = layouts_data["n_turbines"]
    types = layouts_data["types"]
    spacings = layouts_data["spacings"]
    n_layouts = int(layouts_data["n_layouts"])

    # Get unique types and create color mapping
    unique_types = np.unique(types)
    type_colors = dict(zip(unique_types, plt.cm.Set2(np.linspace(0, 1, len(unique_types)))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Layout Distributions (N={n_layouts} layouts)", fontsize=14)

    # 1. Number of turbines histogram - stacked by layout type
    ax = axes[0, 0]
    n_turbines_by_type = [n_turbines[types == t] for t in unique_types]
    colors_list = [type_colors[t] for t in unique_types]
    ax.hist(
        n_turbines_by_type,
        bins=30,
        stacked=True,
        edgecolor="black",
        alpha=0.8,
        color=colors_list,
        label=unique_types,
    )
    ax.axvline(
        np.mean(n_turbines),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(n_turbines):.1f}",
    )
    ax.axvline(
        np.median(n_turbines),
        color="darkred",
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(n_turbines):.1f}",
    )
    ax.set_xlabel("Number of Turbines")
    ax.set_ylabel("Count")
    ax.set_title("Number of Turbines per Layout")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Add text with statistics
    stats_text = (
        f"Min: {np.min(n_turbines)}\nMax: {np.max(n_turbines)}\nStd: {np.std(n_turbines):.1f}"
    )
    ax.text(
        0.98,
        0.55,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
    )

    # 2. Layout types bar chart
    ax = axes[0, 1]
    unique_types_sorted, type_counts = np.unique(types, return_counts=True)
    colors = [type_colors[t] for t in unique_types_sorted]
    bars = ax.bar(unique_types_sorted, type_counts, color=colors, edgecolor="black")
    ax.set_xlabel("Layout Type")
    ax.set_ylabel("Count")
    ax.set_title("Layout Types Distribution")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    # Add count labels on bars
    for bar, count in zip(bars, type_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 3. Spacings histogram - stacked by layout type
    ax = axes[1, 0]
    spacings_by_type = [spacings[types == t] for t in unique_types]
    ax.hist(
        spacings_by_type,
        bins=30,
        stacked=True,
        edgecolor="black",
        alpha=0.8,
        color=colors_list,
        label=unique_types,
    )
    ax.axvline(
        np.mean(spacings),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(spacings):.2f}D",
    )
    ax.axvline(
        np.median(spacings),
        color="darkred",
        linestyle=":",
        linewidth=2,
        label=f"Median: {np.median(spacings):.2f}D",
    )
    ax.set_xlabel("Turbine Spacing (D)")
    ax.set_ylabel("Count")
    ax.set_title("Turbine Spacing Distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    stats_text = (
        f"Min: {np.min(spacings):.2f}D\nMax: {np.max(spacings):.2f}D\nStd: {np.std(spacings):.2f}D"
    )
    ax.text(
        0.98,
        0.55,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
    )

    # 4. Spacing vs N_turbines scatter - colored by layout type
    ax = axes[1, 1]
    for t in unique_types:
        mask = types == t
        ax.scatter(
            spacings[mask],
            n_turbines[mask],
            c=[type_colors[t]],
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label=t,
            s=50,
        )
    ax.set_xlabel("Turbine Spacing (D)")
    ax.set_ylabel("Number of Turbines")
    ax.set_title("Spacing vs Number of Turbines")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "layout_distributions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    return n_turbines, types, spacings


def plot_inflow_distributions(inflows_data: dict, output_dir: Path):
    """Plot distributions of inflow conditions."""
    # Extract inflow data (now a dict)
    wind_speeds = inflows_data["wind_speeds"]
    turbulence_intensities = inflows_data["turbulence_intensities"]
    n_inflows = len(wind_speeds)
    inflows_data.get("n_inflows_per_layout", "?")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Inflow Conditions Distributions (N={n_inflows} inflows)", fontsize=14)

    # 1. Wind speed histogram
    ax = axes[0, 0]
    ax.hist(wind_speeds, bins=30, edgecolor="black", alpha=0.7, color="royalblue")
    ax.axvline(
        np.mean(wind_speeds),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(wind_speeds):.1f} m/s",
    )
    ax.axvline(
        np.median(wind_speeds),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(wind_speeds):.1f} m/s",
    )
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Wind Speed Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    stats_text = f"Min: {np.min(wind_speeds):.1f} m/s\nMax: {np.max(wind_speeds):.1f} m/s\nStd: {np.std(wind_speeds):.1f} m/s"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # 2. Turbulence intensity histogram
    ax = axes[0, 1]
    ax.hist(turbulence_intensities, bins=30, edgecolor="black", alpha=0.7, color="darkorange")
    ax.axvline(
        np.mean(turbulence_intensities),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(turbulence_intensities):.3f}",
    )
    ax.axvline(
        np.median(turbulence_intensities),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(turbulence_intensities):.3f}",
    )
    ax.set_xlabel("Turbulence Intensity (-)")
    ax.set_ylabel("Count")
    ax.set_title("Turbulence Intensity Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    stats_text = f"Min: {np.min(turbulence_intensities):.3f}\nMax: {np.max(turbulence_intensities):.3f}\nStd: {np.std(turbulence_intensities):.3f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # 3. WS vs TI scatter
    ax = axes[1, 0]
    ax.scatter(
        wind_speeds,
        turbulence_intensities,
        alpha=0.4,
        edgecolors="black",
        linewidth=0.3,
        c="purple",
    )
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Turbulence Intensity (-)")
    ax.set_title("Wind Speed vs Turbulence Intensity")
    ax.grid(alpha=0.3)

    # 4. 2D histogram (joint distribution)
    ax = axes[1, 1]
    h = ax.hist2d(wind_speeds, turbulence_intensities, bins=20, cmap="YlOrRd")
    plt.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Turbulence Intensity (-)")
    ax.set_title("Joint Distribution (WS, TI)")

    plt.tight_layout()
    output_path = output_dir / "inflow_distributions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    return wind_speeds, turbulence_intensities


def plot_sample_layouts(layouts_data, output_dir: Path, n_samples: int = 9):
    """Plot a grid of sample layouts."""
    n_layouts = int(layouts_data["n_layouts"])
    n_turbines = layouts_data["n_turbines"]
    types = layouts_data["types"]
    spacings = layouts_data["spacings"]

    # Select diverse samples: smallest, largest, and random
    sorted_indices = np.argsort(n_turbines)
    sample_indices = [
        sorted_indices[0],  # Smallest
        sorted_indices[len(sorted_indices) // 4],  # 25th percentile
        sorted_indices[len(sorted_indices) // 2],  # Median
        sorted_indices[3 * len(sorted_indices) // 4],  # 75th percentile
        sorted_indices[-1],  # Largest
    ]
    # Add random samples
    np.random.seed(42)
    remaining = [i for i in range(n_layouts) if i not in sample_indices]
    sample_indices.extend(np.random.choice(remaining, min(4, len(remaining)), replace=False))
    sample_indices = sample_indices[:n_samples]

    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    fig.suptitle("Sample Layouts", fontsize=14)

    for idx, layout_idx in enumerate(sample_indices):
        ax = axes[idx]
        positions = layouts_data[f"layout_{layout_idx}"]
        n_wt = len(positions)
        layout_type = types[layout_idx]
        spacing = spacings[layout_idx]

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c="red",
            s=80,
            marker="^",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )
        ax.set_title(f"Layout {layout_idx}: {layout_type}\nN={n_wt}, spacing={spacing:.2f}D")
        ax.set_xlabel("x (D)")
        ax.set_ylabel("y (D)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "sample_layouts.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_layout_type_breakdown(layouts_data, output_dir: Path):
    """Plot statistics broken down by layout type."""
    n_turbines = layouts_data["n_turbines"]
    types = layouts_data["types"]
    spacings = layouts_data["spacings"]

    unique_types = np.unique(types)
    n_types = len(unique_types)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Statistics by Layout Type", fontsize=14)

    # Box plot of N turbines by type
    ax = axes[0]
    data_by_type = [n_turbines[types == t] for t in unique_types]
    bp = ax.boxplot(data_by_type, labels=unique_types, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, n_types))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Layout Type")
    ax.set_ylabel("Number of Turbines")
    ax.set_title("Turbines per Layout by Type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    # Box plot of spacings by type
    ax = axes[1]
    data_by_type = [spacings[types == t] for t in unique_types]
    bp = ax.boxplot(data_by_type, labels=unique_types, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Layout Type")
    ax.set_ylabel("Spacing (D)")
    ax.set_title("Spacing by Layout Type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "layout_type_breakdown.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def print_summary(layouts_data, inflows_data: dict):
    """Print summary statistics."""
    n_turbines = layouts_data["n_turbines"]
    types = layouts_data["types"]
    spacings = layouts_data["spacings"]
    n_layouts = int(layouts_data["n_layouts"])

    # inflows_data is now a dict
    wind_speeds = inflows_data["wind_speeds"]
    turbulence_intensities = inflows_data["turbulence_intensities"]
    n_per_layout = inflows_data.get("n_inflows_per_layout", "?")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nLayouts: {n_layouts}")
    print("  Number of turbines:")
    print(f"    Range: {np.min(n_turbines)} - {np.max(n_turbines)}")
    print(f"    Mean ± Std: {np.mean(n_turbines):.1f} ± {np.std(n_turbines):.1f}")
    print(f"    Median: {np.median(n_turbines):.0f}")

    print("\n  Spacing (D):")
    print(f"    Range: {np.min(spacings):.2f} - {np.max(spacings):.2f}")
    print(f"    Mean ± Std: {np.mean(spacings):.2f} ± {np.std(spacings):.2f}")

    print("\n  Layout types:")
    unique_types, counts = np.unique(types, return_counts=True)
    for t, c in zip(unique_types, counts):
        print(f"    {t}: {c} ({100 * c / n_layouts:.1f}%)")

    print(f"\nInflows: {len(wind_speeds)} total ({n_per_layout} per layout)")
    print("  Wind speed (m/s):")
    print(f"    Range: {np.min(wind_speeds):.1f} - {np.max(wind_speeds):.1f}")
    print(f"    Mean ± Std: {np.mean(wind_speeds):.1f} ± {np.std(wind_speeds):.1f}")

    print("\n  Turbulence intensity:")
    print(f"    Range: {np.min(turbulence_intensities):.3f} - {np.max(turbulence_intensities):.3f}")
    print(
        f"    Mean ± Std: {np.mean(turbulence_intensities):.3f} ± {np.std(turbulence_intensities):.3f}"
    )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot dataset distributions")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing npz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: data-dir/distribution_plots)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "distribution_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    print(f"Saving plots to: {output_dir}")

    layouts_data, inflows_data = load_metadata(data_dir)

    # Print summary first
    print_summary(layouts_data, inflows_data)

    # Generate plots
    print("\nGenerating plots...")
    plot_layout_distributions(layouts_data, output_dir)
    plot_inflow_distributions(inflows_data, output_dir)
    plot_sample_layouts(layouts_data, output_dir)
    plot_layout_type_breakdown(layouts_data, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
