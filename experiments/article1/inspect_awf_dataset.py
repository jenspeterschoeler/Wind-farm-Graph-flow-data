"""
Inspect AWF Dataset - visualize layouts, flow fields, and statistics.

This script loads the AWF graph dataset and creates plots for visual inspection:
- Layout geometries with turbine positions
- Flow field contours (unscaled and scaled versions)
- Dataset statistics from multiple flowcases per layout
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from inspection_utils import (
    load_all_graphs_from_zip,
    plot_layout_and_flow,
    plot_multiple_flowcases,
    print_dataset_statistics,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect AWF graph dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/awf_10layouts_test",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--graph-idx",
        type=int,
        default=0,
        help="Graph index to visualize (flowcase index)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/awf_inspection",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--flowcases-per-layout",
        type=int,
        default=4,
        help="Number of flowcases per layout (default: 4 for AWF)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AWF dataset from {data_dir}")

    # AWF stores all graphs in a single zip file
    graphs_zip = data_dir / "graphs.zip"

    if not graphs_zip.exists():
        print(f"Error: {graphs_zip} not found!")
        return

    # Load all graphs using shared utility
    print(f"Loading graphs from {graphs_zip}")
    all_graphs = load_all_graphs_from_zip(graphs_zip)
    print(f"Found {len(all_graphs)} graphs total")

    n_graphs = len(all_graphs)
    n_layouts = n_graphs // args.flowcases_per_layout

    print(f"Dataset contains {n_layouts} layouts with {args.flowcases_per_layout} flowcases each")

    # Plot first layout with all its flowcases
    print(f"\n=== Visualizing Layout 0 (all {args.flowcases_per_layout} flowcases) ===")
    layout_0_graphs = all_graphs[: args.flowcases_per_layout]
    fig = plot_multiple_flowcases(layout_0_graphs, n_cols=2)
    output_path = output_dir / "layout0_all_flowcases.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Plot single flowcase with unscaled/scaled comparison
    print(f"\n=== Single Flowcase Comparison (Flowcase {args.graph_idx}) ===")
    graph_unscaled = all_graphs[args.graph_idx]

    # Create scaled version
    graph_scaled = all_graphs[args.graph_idx]
    ws_inf = graph_scaled.global_features[0].item()
    scaled_flow = graph_scaled.output_features / ws_inf
    graph_scaled_copy = graph_scaled.__class__(**dict(graph_scaled))
    graph_scaled_copy.output_features = scaled_flow
    graph_scaled_copy.global_features = torch.tensor([1.0, graph_scaled.global_features[1].item()])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plot_layout_and_flow(graph_unscaled, title="Unscaled Flow Field", ax=ax1, scaled=False)
    plot_layout_and_flow(graph_scaled_copy, title="Scaled Flow Field (U/U∞)", ax=ax2, scaled=True)
    plt.tight_layout()

    output_path = output_dir / f"flowcase_{args.graph_idx}_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Plot first 4 layouts (first flowcase of each)
    print("\n=== Multi-Layout Overview (First Flowcase of Each Layout) ===")
    n_layouts_to_plot = min(4, n_layouts)
    overview_graphs = [all_graphs[i * args.flowcases_per_layout] for i in range(n_layouts_to_plot)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, graph in enumerate(overview_graphs):
        ws_inf = graph.global_features[0].item()
        ti_inf = graph.global_features[1].item()
        n_wt = len(graph.pos)
        title = f"Layout {idx}: N_WT={n_wt}, WS={ws_inf:.1f} m/s, TI={ti_inf:.3f}"
        plot_layout_and_flow(graph, title=title, ax=axes[idx], scaled=False)

    plt.tight_layout()
    output_path = output_dir / "layouts_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Print dataset statistics using shared utility
    print("\n=== Dataset Statistics ===")
    print_dataset_statistics(all_graphs)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
