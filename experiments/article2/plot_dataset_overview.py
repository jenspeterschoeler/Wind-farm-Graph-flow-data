"""
Generate dataset overview figure for Article 2.

This script creates a figure documenting the turbopark_2500layouts dataset,
showing example layouts, layout type statistics, and inflow condition distributions.
"""

import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Directory setup
_THIS_DIR = Path(__file__).parent
_DATA_GEN_ROOT = _THIS_DIR.parent.parent
_DATA_DIR = _DATA_GEN_ROOT / "data" / "turbopark_2500layouts"

# Dataset generation parameters
TI_MAX = 0.8  # Maximum TI used in Dimitrov_capped method

# Set matplotlib style
plt.style.use("classic")
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["legend.numpoints"] = 1


def load_dataset_metadata():
    """Load layouts, inflows, and split information."""
    with open(_DATA_DIR / "split_info.json") as f:
        split_info = json.load(f)

    n_layouts = split_info["summary"]["total_layouts"]

    layouts_data = np.load(_DATA_DIR / "layouts_metadata.npz", allow_pickle=True)
    layouts = {f"layout_{i}": layouts_data[f"layout_{i}"] for i in range(n_layouts)}

    inflows_data = np.load(_DATA_DIR / "inflows_metadata.npz", allow_pickle=True)
    inflows = {f"inflows_{i}": inflows_data[f"inflows_{i}"] for i in range(n_layouts)}

    return layouts, inflows, split_info


def get_layout_statistics(layouts):
    """Compute statistics about layout sizes."""
    sizes = [len(layouts[f"layout_{i}"]) for i in range(len(layouts))]
    return {
        "sizes": sizes,
        "min": min(sizes),
        "max": max(sizes),
        "mean": np.mean(sizes),
        "median": np.median(sizes),
    }


def get_layout_extent(layout):
    """Get the maximum extent of a layout in D."""
    x_range = layout[:, 0].max() - layout[:, 0].min()
    y_range = layout[:, 1].max() - layout[:, 1].min()
    return max(x_range, y_range)


def select_example_layouts(layouts, split_info, max_extent=180):
    """Select 4 representative layouts that fit within the plot format."""
    all_entries = split_info["train"] + split_info["val"] + split_info["test"]

    by_type = {}
    for entry in all_entries:
        layout_type = entry["type"]
        idx = entry["layout_idx"]
        layout = layouts[f"layout_{idx}"]
        extent = get_layout_extent(layout)

        if extent <= max_extent:
            if layout_type not in by_type:
                by_type[layout_type] = []
            by_type[layout_type].append((entry, extent))

    type_names = {
        "cluster": "Cluster",
        "single string": "Single string",
        "parallel string": "Parallel string",
        "multiple string": "Multiple string",
    }

    selected = []
    for layout_type in ["cluster", "single string", "parallel string", "multiple string"]:
        if layout_type in by_type:
            entries = by_type[layout_type]
            entries_sorted = sorted(entries, key=lambda x: abs(x[1] - 90))

            best_entry = None
            for entry, _extent in entries_sorted[:20]:
                n_wt = entry["n_turbines"]
                if 10 <= n_wt <= 50:
                    best_entry = entry
                    break
            if best_entry is None:
                best_entry = entries_sorted[0][0]

            idx = best_entry["layout_idx"]
            title = type_names.get(layout_type, layout_type.title())
            selected.append((layouts[f"layout_{idx}"], title, idx))

    return selected


def collect_inflows_by_split(inflows, split_info):
    """Collect all inflow conditions grouped by train/val/test split."""
    train_inflows = []
    val_inflows = []
    test_inflows = []

    for entry in split_info["train"]:
        train_inflows.append(inflows[f"inflows_{entry['layout_idx']}"])
    for entry in split_info["val"]:
        val_inflows.append(inflows[f"inflows_{entry['layout_idx']}"])
    for entry in split_info["test"]:
        test_inflows.append(inflows[f"inflows_{entry['layout_idx']}"])

    return np.vstack(train_inflows), np.vstack(val_inflows), np.vstack(test_inflows)


def collect_layout_stats_by_type(split_info):
    """Collect layout statistics grouped by type."""
    all_entries = split_info["train"] + split_info["val"] + split_info["test"]

    by_type = {}
    for entry in all_entries:
        layout_type = entry["type"]
        if layout_type not in by_type:
            by_type[layout_type] = {"n_turbines": [], "spacing": []}
        by_type[layout_type]["n_turbines"].append(entry["n_turbines"])
        by_type[layout_type]["spacing"].append(entry["spacing"])

    return by_type


def main():
    """Generate the dataset overview figure."""
    print("Loading dataset metadata...")
    layouts, inflows, split_info = load_dataset_metadata()

    n_layouts = len(layouts)
    stats = get_layout_statistics(layouts)
    print("\nDataset Summary:")
    print(f"  Total layouts: {n_layouts}")
    print(f"  Train/Val/Test: {split_info['summary']}")
    print(
        f"  Turbines per layout: min={stats['min']}, max={stats['max']}, "
        f"mean={stats['mean']:.1f}, median={stats['median']:.0f}"
    )

    # Collect data
    train_inflows, val_inflows, test_inflows = collect_inflows_by_split(inflows, split_info)
    layout_stats_by_type = collect_layout_stats_by_type(split_info)
    example_layouts = select_example_layouts(layouts, split_info)

    # Extract wind speed and TI
    u_split = [train_inflows[:, 0], val_inflows[:, 0], test_inflows[:, 0]]
    ti_split = [train_inflows[:, 1], val_inflows[:, 1], test_inflows[:, 1]]

    all_u = np.concatenate(u_split)
    all_ti = np.concatenate(ti_split)
    cutin = all_u.min()
    cutout = all_u.max()

    # IEC distribution boundaries with TI_MAX cap
    u_lin = np.linspace(cutin, cutout, 100)
    I_refAp = 0.18
    sigma_upper_uncapped = I_refAp * (6.8 + 0.75 * u_lin + 3 * (10 / u_lin) ** 2)
    sigma_lower = 0.0025 * u_lin
    sigma_upper = np.minimum(sigma_upper_uncapped, TI_MAX * u_lin)
    TI_upper = sigma_upper / u_lin
    TI_lower = sigma_lower / u_lin

    # Figure setup
    figure_ids = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
    id_pos = (0.9, 0.9)

    # Data split colors from buda_r (reversed, continuous colormap, sampled at distinct positions)
    split_colors = [cmc.buda_r(i) for i in [0.1, 0.5, 0.9]]
    markers = ["o", "s", "^"]
    alpha = 0.8

    # Layout type colors from batlowS (blue-green-yellow range)
    type_colors = {
        "cluster": cmc.batlowS(0),
        "single string": cmc.batlowS(3),
        "parallel string": cmc.batlowS(5),
        "multiple string": cmc.batlowS(7),
    }
    type_order = ["cluster", "single string", "parallel string", "multiple string"]
    type_labels = ["Cluster", "Single str.", "Parallel str.", "Multiple str."]

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(12, 8.5))

    # Define grid: 3 rows
    # Row 0: 4 equal columns for layouts
    # Row 1: 3 columns for boxplots (wider) + 1 column for bar chart
    # Row 2: 4 equal columns for inflow stats
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1, 1],
        hspace=0.35,
        wspace=0.3,
    )

    # ===== TOP ROW: Example layouts (shared y-axis) =====
    upper_axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, 4):
        upper_axes.append(fig.add_subplot(gs[0, i], sharey=upper_axes[0]))

    for i, (ax, (layout, title, _idx)) in enumerate(zip(upper_axes, example_layouts)):
        n_wt = len(layout)
        ax.scatter(layout[:, 0], layout[:, 1], marker="2", alpha=0.8, color="black", s=40)
        ax.set_xlabel(r"$x/D$ [-]")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        ax.text(
            *id_pos,
            figure_ids[i],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.05,
            0.92,
            f"{title}",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.text(
            0.05,
            0.80,
            rf"$n_{{\mathrm{{wt}}}}={n_wt}$",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    upper_axes[0].set_ylabel(r"$y/D$ [-]")
    # Hide y-tick labels for shared axes
    for ax in upper_axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # ===== MIDDLE ROW: Boxplots (spanning 3 cols) + Bar chart =====
    # Use nested gridspec for middle row to have different widths
    gs_middle = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, :], width_ratios=[1.5, 1.5, 1], wspace=0.15
    )

    # (e) Number of turbines boxplot (horizontal)
    ax_box1 = fig.add_subplot(gs_middle[0])
    n_turbines_data = [layout_stats_by_type[t]["n_turbines"] for t in type_order]
    bp1 = ax_box1.boxplot(
        n_turbines_data,
        vert=False,
        tick_labels=type_labels,
        patch_artist=True,
    )
    for patch, t in zip(bp1["boxes"], type_order):
        patch.set_facecolor(type_colors[t])
        patch.set_edgecolor("black")
        patch.set_alpha(0.7)
    for whisker in bp1["whiskers"]:
        whisker.set_color("black")
        whisker.set_linestyle("-")
    for cap in bp1["caps"]:
        cap.set_color("black")
    for median in bp1["medians"]:
        median.set_color("black")
    ax_box1.set_xlabel(r"Number of turbines [-]")
    ax_box1.invert_yaxis()  # Invert so Cluster is at top
    ax_box1.set_xlim(0, 320)  # Extend upper limit to see outliers
    ax_box1.text(
        0.95,
        0.1,  # Moved to bottom since y-axis is inverted
        figure_ids[4],
        ha="center",
        va="center",
        transform=ax_box1.transAxes,
    )

    # (f) Spacing boxplot (horizontal) - share y-axis labels with first boxplot
    ax_box2 = fig.add_subplot(gs_middle[1])
    spacing_data = [layout_stats_by_type[t]["spacing"] for t in type_order]
    bp2 = ax_box2.boxplot(
        spacing_data,
        vert=False,
        tick_labels=[""] * len(type_labels),  # Empty labels, shared with ax_box1
        patch_artist=True,
    )
    for patch, t in zip(bp2["boxes"], type_order):
        patch.set_facecolor(type_colors[t])
        patch.set_edgecolor("black")
        patch.set_alpha(0.7)
    for whisker in bp2["whiskers"]:
        whisker.set_color("black")
        whisker.set_linestyle("-")
    for cap in bp2["caps"]:
        cap.set_color("black")
    for median in bp2["medians"]:
        median.set_color("black")
    ax_box2.set_xlabel(r"$s_\mathrm{wt}/D$ [-]")
    ax_box2.invert_yaxis()  # Invert so Cluster is at top
    ax_box2.set_xlim(2, 11)  # Extend both ends to see boxplot ends
    ax_box2.text(
        0.95,
        0.1,  # Moved to bottom since y-axis is inverted
        figure_ids[5],
        ha="center",
        va="center",
        transform=ax_box2.transAxes,
    )

    # (g) Layout type count bar chart - share y-axis labels with boxplots
    ax_bar = fig.add_subplot(gs_middle[2])
    type_counts = [len(layout_stats_by_type[t]["n_turbines"]) for t in type_order]
    bars = ax_bar.barh(
        type_labels,
        type_counts,
        color=[type_colors[t] for t in type_order],
        alpha=0.8,
        edgecolor="black",
    )
    ax_bar.set_xlabel(r"Number of layouts [-]")
    ax_bar.set_yticklabels([])  # Empty labels, shared with ax_box1
    ax_bar.invert_yaxis()  # Invert so Cluster is at top
    ax_bar.set_xlim(0, 1100)  # Extend x-axis to fit bars and labels
    ax_bar.set_xticks([0, 500, 1000])  # Reduce number of x-ticks
    ax_bar.text(
        0.9,
        0.1,  # Moved to bottom since y-axis is inverted
        figure_ids[6],
        ha="center",
        va="center",
        transform=ax_bar.transAxes,
    )
    # Add count labels
    for bar, count in zip(bars, type_counts):
        ax_bar.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            ha="left",
            va="center",
            fontsize=9,
        )

    # ===== BOTTOM ROW: Inflow statistics =====
    # Use nested gridspec for bottom row with 5 columns (small spacer between h and i, larger before j)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs[2, :], width_ratios=[1, 0.15, 1, 0.35, 2.2], wspace=0.12
    )
    # Axes at positions 0 (h) and 2 (i), skipping spacer at position 1
    lower_axes = [fig.add_subplot(gs_bottom[0]), fig.add_subplot(gs_bottom[2])]

    # Convert TI to percent
    ti_split_pct = [ti * 100 for ti in ti_split]
    all_ti_pct = all_ti * 100

    # (h) Wind speed histogram
    ws_bins = int((cutout - cutin) / 1.5)
    lower_axes[0].hist(
        u_split,
        ws_bins,
        histtype="bar",
        stacked=True,
        color=split_colors,
        alpha=alpha,
    )
    lower_axes[0].text(
        *id_pos,
        figure_ids[7],
        ha="center",
        va="center",
        transform=lower_axes[0].transAxes,
    )
    lower_axes[0].set_xlabel(r"$U\ [\mathrm{ms}^{-1}]$")
    lower_axes[0].set_ylabel(r"Number of samples [-]")

    # (i) TI histogram (in percent) - share y-axis with wind speed histogram
    ti_bins = int((all_ti_pct.max() - all_ti_pct.min()) / 5)  # Wider bins to match wind speed
    lower_axes[1].hist(
        ti_split_pct,
        ti_bins,
        histtype="bar",
        stacked=True,
        color=split_colors,
        alpha=alpha,
    )
    lower_axes[1].text(
        *id_pos,
        figure_ids[8],
        ha="center",
        va="center",
        transform=lower_axes[1].transAxes,
    )
    lower_axes[1].set_xlabel(r"$I_0$ [%]")

    # Share y-axis limits and remove y-ticks from (i)
    y_max = max(lower_axes[0].get_ylim()[1], lower_axes[1].get_ylim()[1])
    lower_axes[0].set_ylim(0, y_max)
    lower_axes[1].set_ylim(0, y_max)
    lower_axes[1].set_yticklabels([])  # Remove y-tick labels, shared with (h)

    # (j) U vs TI scatter with capped distribution boundary (TI in percent)
    combined_u = np.concatenate(u_split)
    combined_ti_pct = np.concatenate(ti_split_pct)
    labels_u = np.concatenate([np.full(len(s), i) for i, s in enumerate(u_split)])

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(labels_u))
    downsampling_factor = max(1, len(perm) // 5000)
    perm = perm[::downsampling_factor]

    u_shuf = combined_u[perm]
    ti_shuf_pct = combined_ti_pct[perm]
    labels_shuf = labels_u[perm]

    # Scatter plot in fifth column of bottom row (positions 1 and 3 are spacers)
    ax_scatter = fig.add_subplot(gs_bottom[4])

    for start in range(0, len(labels_shuf), 1000):
        end = start + 1000
        for cls in range(len(split_colors)):
            mask = labels_shuf[start:end] == cls
            if mask.any():
                ax_scatter.scatter(
                    u_shuf[start:end][mask],
                    ti_shuf_pct[start:end][mask],
                    color=split_colors[cls],
                    marker=markers[cls],
                    alpha=alpha,
                    s=10,
                    edgecolor="k",
                    linewidths=0.5,
                )

    # Plot capped IEC distribution boundaries (in percent)
    TI_upper_pct = TI_upper * 100
    TI_lower_pct = TI_lower * 100
    ax_scatter.plot(u_lin, TI_upper_pct, color="black", linestyle="-", linewidth=1)
    ax_scatter.plot(u_lin, TI_lower_pct, color="black", linestyle="-", linewidth=1)
    ax_scatter.plot(
        [cutin, cutin],
        [TI_lower_pct[0], TI_upper_pct[0]],
        color="black",
        linestyle="-",
        linewidth=1,
    )
    ax_scatter.plot(
        [cutout, cutout],
        [TI_lower_pct[-1], TI_upper_pct[-1]],
        color="black",
        linestyle="-",
        linewidth=1,
    )

    ax_scatter.text(
        0.95,
        0.9,
        figure_ids[9],
        ha="center",
        va="center",
        transform=ax_scatter.transAxes,
    )
    ax_scatter.set_xlabel(r"$U\ [\mathrm{ms}^{-1}]$")
    ax_scatter.set_ylabel(r"$I_0$ [%]")

    # Create legend with three rows: other items first, then layout types, then data splits
    # Note: matplotlib fills legends by column, so we interleave items for row-wise appearance
    marker_size = 10
    title_fontsize = 13
    entry_fontsize = 11

    # Row 1: Wind turbine, Distribution boundary, (empty x3)
    # Row 2: Layout types header, Cluster, Single, Parallel, Multiple
    # Row 3: Data splits header, Training, Validation, Test, (empty)
    # With ncol=5, we need 15 items total for 3 complete rows
    # Interleaved order for column-wise filling:

    legend_elements = [
        # Column 1: Wind turbine, Layout types header, Data splits header
        Line2D(
            [0],
            [0],
            marker="2",
            color="w",
            label="Wind turbine",
            markeredgecolor="k",
            markerfacecolor="k",
            markeredgewidth=2,
            lw=0.000001,
            markersize=marker_size,
        ),
        Line2D([0], [0], color="w", label=r"$\bf{Layout\ types:}$", lw=0),
        Line2D([0], [0], color="w", label=r"$\bf{Data\ splits:}$", lw=0),
        # Column 2: Distribution boundary, Cluster, Training
        Line2D([0], [0], color="k", label="Distribution boundary", lw=1),
        Patch(facecolor=type_colors["cluster"], edgecolor="black", label="Cluster"),
        Line2D(
            [0],
            [0],
            marker=markers[0],
            color="w",
            label="Training",
            markerfacecolor=split_colors[0],
            markersize=marker_size,
        ),
        # Column 3: empty, Single string, Validation
        Line2D([0], [0], color="w", label=" ", lw=0),
        Patch(facecolor=type_colors["single string"], edgecolor="black", label="Single string"),
        Line2D(
            [0],
            [0],
            marker=markers[1],
            color="w",
            label="Validation",
            markerfacecolor=split_colors[1],
            markersize=marker_size,
        ),
        # Column 4: empty, Parallel string, Test
        Line2D([0], [0], color="w", label=" ", lw=0),
        Patch(facecolor=type_colors["parallel string"], edgecolor="black", label="Parallel string"),
        Line2D(
            [0],
            [0],
            marker=markers[2],
            color="w",
            label="Test",
            markerfacecolor=split_colors[2],
            markersize=marker_size,
        ),
        # Column 5: empty, Multiple string, empty
        Line2D([0], [0], color="w", label=" ", lw=0),
        Patch(facecolor=type_colors["multiple string"], edgecolor="black", label="Multiple string"),
        Line2D([0], [0], color="w", label=" ", lw=0),
    ]

    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.51, -0.06),  # Aligned with plot areas
        ncol=5,
        borderaxespad=0.0,
        columnspacing=2.0,  # Increased spacing to spread across plot width
        handletextpad=0.5,
        borderpad=0.3,
        fontsize=entry_fontsize,
    )
    # Make title entries larger
    for text in leg.get_texts():
        if "Layout types" in text.get_text() or "Data splits" in text.get_text():
            text.set_fontsize(title_fontsize)

    # Save figure
    output_path = _THIS_DIR / "figures" / "dataset_overview.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"PNG saved to: {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
