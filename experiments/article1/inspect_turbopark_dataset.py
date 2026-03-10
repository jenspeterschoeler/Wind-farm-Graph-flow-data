"""
Inspect TurbOPark dataset - visualize layouts, flow fields, and statistics.

This script loads the generated dataset and creates plots for visual inspection:
- Layout geometries with turbine positions
- Flow field contours (unscaled and scaled versions)
- Dataset statistics
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inspection_utils import load_graph_from_zip, plot_layout_and_flow


def plot_dataset_statistics(stats_path: Path):
    """Plot dataset statistics from stats.json."""
    with open(stats_path) as f:
        stats = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Dataset Statistics", fontsize=16)

    # Node features (turbine wind speeds)
    ax = axes[0, 0]
    node_stats = stats["node_features"]
    ax.bar(
        ["Mean", "Std", "Min", "Max"],
        [node_stats["mean"][0], node_stats["std"][0], node_stats["min"], node_stats["max"]],
    )
    ax.set_title("Node Features (Turbine Wind Speed)")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(alpha=0.3)

    # Global features (inflow conditions)
    ax = axes[0, 1]
    global_stats = stats["global_features"]
    x_pos = np.arange(4)
    ws_vals = [
        global_stats["mean"][0],
        global_stats["std"][0],
        global_stats["min"][0],
        global_stats["max"][0],
    ]
    ax.bar(x_pos, ws_vals, color="blue", alpha=0.7, label="Wind Speed")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Mean", "Std", "Min", "Max"])
    ax.set_title("Global Features - Wind Speed")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Turbulence intensity
    ax = axes[0, 2]
    ti_vals = [
        global_stats["mean"][1],
        global_stats["std"][1],
        global_stats["min"][1],
        global_stats["max"][1],
    ]
    ax.bar(x_pos, ti_vals, color="orange", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Mean", "Std", "Min", "Max"])
    ax.set_title("Global Features - Turbulence Intensity")
    ax.set_ylabel("TI (-)")
    ax.grid(alpha=0.3)

    # Output features (flow field velocities)
    ax = axes[1, 0]
    output_stats = stats["output"]
    ax.bar(
        ["Mean", "Std", "Min", "Max"],
        [output_stats["mean"][0], output_stats["std"][0], output_stats["min"], output_stats["max"]],
        color="green",
        alpha=0.7,
    )
    ax.set_title("Output Features (Flow Field Velocity)")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(alpha=0.3)

    # Graph sizes
    ax = axes[1, 1]
    graph_size = stats["graph_size"]
    ax.bar(
        ["Nodes", "Edges", "Globals"],
        [graph_size["max_n_nodes"], graph_size["max_n_edges"], graph_size["max_n_globals"]],
        color="purple",
        alpha=0.7,
    )
    ax.set_title("Maximum Graph Sizes")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

    # Dataset info
    ax = axes[1, 2]
    ax.text(
        0.1,
        0.7,
        f"Total Graphs: {stats['n_graphs']}\n"
        f"Trunk Shape: {stats['trunk_shape']}\n"
        f"Date: {stats['date_ISO8601']}",
        fontsize=12,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title("Dataset Info")
    ax.axis("off")

    plt.tight_layout()
    return fig


def main():
    """Main inspection function."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect TurbOPark dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/turbopark_10layouts_test",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--layout-idx",
        type=int,
        default=0,
        help="Layout index to visualize (default: 0)",
    )
    parser.add_argument(
        "--graph-idx",
        type=int,
        default=0,
        help="Graph index within layout to visualize (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/turbopark_inspection",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting dataset: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Find all layout zips (unscaled data)
    layout_zips = sorted(data_dir.glob("_layout*.zip"))
    train_zips = sorted((data_dir / "train_pre_processed").glob("_layout*.zip"))

    if not layout_zips:
        print(f"ERROR: No layout zip files found in {data_dir}")
        return

    print(f"Found {len(layout_zips)} layouts")

    # Load statistics
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        print("Plotting dataset statistics...")
        fig_stats = plot_dataset_statistics(stats_path)
        stats_output = output_dir / "dataset_statistics.png"
        fig_stats.savefig(stats_output, dpi=150, bbox_inches="tight")
        print(f"Saved: {stats_output}")
        plt.close(fig_stats)

    # Select layout to visualize
    layout_idx = min(args.layout_idx, len(layout_zips) - 1)
    layout_zip = layout_zips[layout_idx]

    print(f"\nVisualizing layout {layout_idx}: {layout_zip.name}")

    # Load unscaled graph using shared utility
    graph_unscaled = load_graph_from_zip(layout_zip, args.graph_idx)

    # Create figure with unscaled and scaled plots
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    # Plot unscaled data
    plot_layout_and_flow(
        graph_unscaled,
        title=f"Flow Field (Unscaled) - Layout {layout_idx}",
        ax=axes[0],
        scaled=False,
    )

    # Try to load scaled version
    if train_zips:
        # Find corresponding scaled layout
        layout_name = layout_zip.stem  # e.g., "_layout0"
        scaled_zip = None
        for z in train_zips:
            if z.stem == layout_name:
                scaled_zip = z
                break

        if scaled_zip:
            graph_scaled = load_graph_from_zip(scaled_zip, args.graph_idx)
            plot_layout_and_flow(
                graph_scaled,
                title=f"Flow Field (Scaled) - Layout {layout_idx}",
                ax=axes[1],
                scaled=True,
            )
            print(f"Loaded scaled version from: {scaled_zip}")
        else:
            axes[1].text(
                0.5,
                0.5,
                "Scaled version not found in train set",
                ha="center",
                va="center",
                fontsize=14,
            )
            axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "No preprocessed data found", ha="center", va="center", fontsize=14)
        axes[1].axis("off")

    plt.tight_layout()
    flow_output = output_dir / f"flow_field_layout{layout_idx}_graph{args.graph_idx}.png"
    fig.savefig(flow_output, dpi=150, bbox_inches="tight")
    print(f"Saved: {flow_output}")
    plt.close(fig)

    # Create a multi-layout overview (first 4 layouts)
    n_overview = min(4, len(layout_zips))
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    print(f"\nCreating overview of first {n_overview} layouts...")
    for i in range(n_overview):
        graph = load_graph_from_zip(layout_zips[i], 0)
        plot_layout_and_flow(graph, title=f"Layout {i} (Unscaled)", ax=axes[i], scaled=False)

    plt.tight_layout()
    overview_output = output_dir / "layouts_overview.png"
    fig.savefig(overview_output, dpi=150, bbox_inches="tight")
    print(f"Saved: {overview_output}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Inspection complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
