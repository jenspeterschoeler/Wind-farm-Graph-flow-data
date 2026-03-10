"""
Shared utilities for dataset inspection scripts.

This module provides common functions for loading graphs from zip files
and plotting flow fields, used by both AWF and TurbOPark inspection scripts.
"""

from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_graph_from_zip(zip_path: Path, graph_idx: int = 0):
    """
    Load a single graph from a zip file.

    Args:
        zip_path: Path to the zip file containing graph .pt files
        graph_idx: Index of the graph to load (default: 0)

    Returns:
        PyTorch Geometric Data object
    """
    with ZipFile(zip_path, "r") as zf:
        filenames = sorted([f for f in zf.namelist() if f.endswith(".pt")])
        if graph_idx >= len(filenames):
            graph_idx = 0
        with zf.open(filenames[graph_idx]) as f:
            # PyTorch 2.6+ requires weights_only=False for custom classes
            graph = torch.load(f, weights_only=False)
    return graph


def load_all_graphs_from_zip(zip_path: Path):
    """
    Load all graphs from a zip file.

    Args:
        zip_path: Path to the zip file containing graph .pt files

    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    with ZipFile(zip_path, "r") as zf:
        filenames = sorted([f for f in zf.namelist() if f.endswith(".pt")])
        for filename in filenames:
            with zf.open(filename) as f:
                graph = torch.load(f, weights_only=False)
                graphs.append(graph)
    return graphs


def plot_layout_and_flow(graph, title="Flow Field", ax=None, scaled=False):
    """
    Plot turbine layout and flow field.

    Args:
        graph: PyTorch Geometric Data object with:
            - pos: Turbine positions [N, 2]
            - trunk_inputs: Probe positions [M, 2]
            - output_features: Flow field values [M, 1]
            - global_features: [wind_speed, turbulence_intensity]
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        scaled: Whether data is scaled (affects colorbar label)

    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data - PyGTupleData uses pos for positions, not x
    wt_positions = graph.pos.numpy()  # Turbine positions (x, y)
    probe_positions = graph.trunk_inputs.numpy()  # Probe positions (x, y)
    flow_velocities = graph.output_features.numpy().squeeze()  # Flow field values
    ws_inf = graph.global_features[0].item()
    ti_inf = graph.global_features[1].item()

    # Reshape flow field to grid
    n_x = len(np.unique(probe_positions[:, 0]))
    n_y = len(np.unique(probe_positions[:, 1]))

    # Sort by x then y to get proper grid ordering
    sort_idx = np.lexsort((probe_positions[:, 1], probe_positions[:, 0]))
    probe_positions_sorted = probe_positions[sort_idx]
    flow_velocities_sorted = flow_velocities[sort_idx]

    # Reshape to grid
    x_grid = probe_positions_sorted[:, 0].reshape(n_x, n_y)
    y_grid = probe_positions_sorted[:, 1].reshape(n_x, n_y)
    flow_grid = flow_velocities_sorted.reshape(n_x, n_y)

    # Plot flow field
    velocity_label = "Velocity (scaled)" if scaled else "Velocity (m/s)"
    contour = ax.contourf(x_grid, y_grid, flow_grid, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax, label=velocity_label)

    # Plot turbines
    ax.scatter(
        wt_positions[:, 0],
        wt_positions[:, 1],
        c="red",
        s=100,
        marker="^",
        edgecolors="white",
        linewidths=1.5,
        label="Wind Turbines",
        zorder=10,
    )

    # Add title with metadata
    ws_label = f"WS∞={ws_inf:.1f}"
    if not scaled:
        ws_label += " m/s"
    ax.set_title(f"{title}\n{ws_label}, TI∞={ti_inf:.3f}, N_WT={len(wt_positions)}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    return ax


def plot_multiple_flowcases(graphs, n_cols=2):
    """
    Plot multiple flowcases from same layout in a grid.

    Args:
        graphs: List of graph objects (same layout, different flowcases)
        n_cols: Number of columns in the grid

    Returns:
        Matplotlib figure object
    """
    n_graphs = len(graphs)
    n_rows = (n_graphs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, graph in enumerate(graphs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        ws_inf = graph.global_features[0].item()
        ti_inf = graph.global_features[1].item()
        title = f"Flowcase {idx}: WS={ws_inf:.1f} m/s, TI={ti_inf:.3f}"
        plot_layout_and_flow(graph, title=title, ax=ax, scaled=False)

    # Hide empty subplots
    for idx in range(n_graphs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def print_dataset_statistics(graphs):
    """
    Print summary statistics for a list of graphs.

    Args:
        graphs: List of PyTorch Geometric Data objects
    """
    all_ws = [g.global_features[0].item() for g in graphs]
    all_ti = [g.global_features[1].item() for g in graphs]
    all_nwt = [len(g.pos) for g in graphs]

    print("Wind Speed (m/s):")
    print(f"  Range: {min(all_ws):.1f} - {max(all_ws):.1f}")
    print(f"  Mean: {np.mean(all_ws):.1f} ± {np.std(all_ws):.1f}")

    print("\nTurbulence Intensity:")
    print(f"  Range: {min(all_ti):.3f} - {max(all_ti):.3f}")
    print(f"  Mean: {np.mean(all_ti):.3f} ± {np.std(all_ti):.3f}")

    print("\nNumber of Turbines per Layout:")
    print(f"  Range: {min(all_nwt)} - {max(all_nwt)}")
    print(f"  Mean: {np.mean(all_nwt):.1f} ± {np.std(all_nwt):.1f}")
