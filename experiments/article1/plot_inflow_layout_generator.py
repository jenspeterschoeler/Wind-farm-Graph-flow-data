"""Inflow and layout plotting for Article 1 experiments."""

# Add main path to sys path
import sys
from pathlib import Path

# Get the data-generation root directory (two levels up from article1/)
_THIS_DIR = Path(__file__).parent
_DATA_GEN_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_DATA_GEN_ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from py_wake.examples.data.dtu10mw import DTU10MW  # noqa: E402
from py_wake.examples.data.dtu10mw import power_curve as power_curve_dtu10mw  # noqa: E402

from utils.inflow_generator import (  # noqa: E402
    IEC_61400_1_2019_class_interpreter,
    InflowGenerator,
)
from utils.layout_generator import (  # noqa: E402
    PLayGen,
    layout_limits_run1,
    setup_eval_layout,
)

# set random seeds for reproducibility
np.random.seed(8)

# set style to classic, but remove graybackground
plt.style.use("classic")
matplotlib.rcParams["axes.facecolor"] = "white"
# set font size
matplotlib.rcParams.update({"font.size": 12})
matplotlib.rcParams["legend.numpoints"] = 1


eval_layout = setup_eval_layout(layout_limits_run1)

layout_generator = PLayGen(D=1)

keep_updating_cluster = True
keep_updating_single_string = True
keep_updating_multiple_string = True
keep_updating_parallel_string = True

while (
    keep_updating_cluster
    or keep_updating_single_string
    or keep_updating_multiple_string
    or keep_updating_parallel_string
):
    cluster_layout = layout_generator.random_cluster_layout()
    string_layout = layout_generator.random_single_string_layout()
    multiple_string_layout = layout_generator.random_multiple_string_layout()
    parallel_string_layout = layout_generator.random_parallel_string_layout()

    if (
        eval_layout(
            cluster_layout,
            layout_generator._interturbine_spacing_(cluster_layout[:, 0], cluster_layout[:, 1]),
        )
        and keep_updating_cluster
    ):
        keep_updating_cluster = False
        cluster_layout_final = cluster_layout
        print("Cluster layout found")

    if (
        eval_layout(
            string_layout,
            layout_generator._interturbine_spacing_(string_layout[:, 0], string_layout[:, 1]),
        )
        and keep_updating_single_string
    ):
        keep_updating_single_string = False
        string_layout_final = string_layout
        print("Single string layout found")

    if (
        eval_layout(
            multiple_string_layout,
            layout_generator._interturbine_spacing_(
                multiple_string_layout[:, 0], multiple_string_layout[:, 1]
            ),
        )
        and keep_updating_multiple_string
    ):
        keep_updating_multiple_string = False
        multiple_string_layout_final = multiple_string_layout
        print("Multiple string layout found")

    if (
        eval_layout(
            parallel_string_layout,
            layout_generator._interturbine_spacing_(
                parallel_string_layout[:, 0], parallel_string_layout[:, 1]
            ),
        )
        and keep_updating_parallel_string
    ):
        keep_updating_parallel_string = False
        parallel_string_layout_final = parallel_string_layout
        print("Parallel string layout found")


wt = DTU10MW()
D = wt.diameter()


hub_height = wt.hub_height()
cutin = power_curve_dtu10mw[:, 0].min()
cutout = power_curve_dtu10mw[:, 0].max()

inflow_settings = IEC_61400_1_2019_class_interpreter(wt_class="I", ti_charataristics="B")

turbine_settings = {
    "cutin_u": cutin,
    "cutout_u": cutout,
    "height_above_ground": wt.hub_height(),
}

inflow_gen = InflowGenerator(inflow_settings=inflow_settings, turbine_settings=turbine_settings)
# num_samples = int(50000/100)
# train = int(0.6 * num_samples)
# test = int(0.2 * num_samples)

num_samples = 3570 * 10
train = 2072 * 10
test = 998 * 10

samples = inflow_gen.sampler.random(n=num_samples)
u = inflow_gen._gen_wind_velocities(samples[:, 0])
ti = inflow_gen._gen_turbulence(u, samples[:, 1])

samples_split = [
    samples[:train],
    samples[train : train + test],
    samples[train + test :],
]
u_split = [u[:train], u[train + test :], u[train : train + test]]
ti_split = [
    ti[:train],
    ti[train + test :],
    ti[train : train + test],
]

u_lin = np.linspace(cutin, cutout, 100)
I_refAp = 0.18  # The IEC 61400-1 reference turbulence intensity for A+
# Upper and lower bounds below from Table 1 in Dimitrov et al. (2018)
sigma_upper = I_refAp * (6.8 + 0.75 * u_lin + 3 * (10 / u_lin) ** 2)
sigma_lower = 0.0025 * u_lin

TI_upper = sigma_upper / u_lin
TI_lower = sigma_lower / u_lin


figure_ids = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
id_pos = (0.9, 0.9)
colors = ["blue", "gold", "red"]
markers = ["o", "s", "^"]
alpha = 0.8


figure, axes = plt.subplots(2, 4, figsize=(12, 5.75))
upper_axes = axes[0, :]
lower_axes = axes[1, :]


for i, (ax, layout, title) in enumerate(
    zip(
        upper_axes,
        [
            cluster_layout_final,
            string_layout_final,
            parallel_string_layout_final,
            multiple_string_layout_final,
        ],
        ["Cluster", "Single string", "Parallel string", "Multiple string"],
    )
):
    layout = layout
    ax.scatter(layout[:, 0], layout[:, 1], marker="2", alpha=0.8, color="black", s=40)
    # ax.axis("equal")
    ax.set_xlabel(r"$x/D$ [-]")
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    ax.text(
        *id_pos,
        figure_ids[i],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.9,
        title,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )

upper_axes[0].set_ylabel(r"$y/D$ [-]")


# for samples_, color, marker_type in zip(samples_split, colors, markers):
#     lower_axes[0].scatter(samples_[:, 0], samples_[:, 1], color=color, alpha=alpha, marker=marker_type)

combined = np.vstack(samples_split)
labels = np.concatenate(
    [np.full(len(s), i) for i, s in enumerate(samples_split)]
)  # 0=train,1=val,2=test
rng = np.random.default_rng(
    0
)  # fixed seed for reproducible plotting; remove seed for different shuffle each run
downsampling_factor = 10

perm = rng.permutation(len(combined))
perm = perm[::downsampling_factor]
combined_shuf = combined[perm]
labels_shuf = labels[perm]
color_map = np.array(colors)[labels_shuf]
# markers = np.array(markers)[labels_shuf]

# # single scatter with mixed order (single marker for all points)
# lower_axes[0].scatter(combined_shuf[:, 0], combined_shuf[:, 1], c=color_map, alpha=alpha, marker='o', s=10)
# preserve per-class markers/colors while interleaving points
colors_arr = np.array(colors)
markers_arr = np.array(markers)
chunk_size = 2000  # smaller -> more interleaving but slower, tune as needed
for start in range(0, len(labels_shuf), chunk_size):
    end = start + chunk_size
    for cls in range(len(colors_arr)):
        mask = labels_shuf[start:end] == cls
        if mask.any():
            pts = combined_shuf[start:end][mask]
            lower_axes[0].scatter(
                pts[:, 0],
                pts[:, 1],
                color=colors_arr[cls],
                marker=markers_arr[cls],
                edgecolor="k",
                linewidths=0.5,
                alpha=alpha,
                s=10,
            )


# lower_axes[0].set_title("Sobol samples")
lower_axes[0].text(
    *id_pos,
    figure_ids[4],
    horizontalalignment="center",
    verticalalignment="center",
    transform=lower_axes[0].transAxes,
)
lower_axes[0].set_xlabel("$p_{U}$ [-]")
lower_axes[0].set_ylabel(r"$p_\mathrm{TI}$ [-]")
lower_axes[0].tick_params(axis="x", rotation=45)  # rotate x-axis ticks for better readability
lower_axes[0].xaxis.set_label_coords(0.5, -0.2)


lower_axes[1].hist(
    u_split,
    int((cutout - cutin) / 2),
    histtype="bar",
    stacked=True,
    color=colors,
    alpha=alpha,
)
# lower_axes[1].set_title("Wind speed")
lower_axes[1].text(
    *id_pos,
    figure_ids[5],
    horizontalalignment="center",
    verticalalignment="center",
    transform=lower_axes[1].transAxes,
)
lower_axes[1].set_xlabel(r"$U\ [\mathrm{ms}^{-1}]$")
lower_axes[1].set_ylabel(r"Number of samples [-]")


lower_axes[2].hist(
    ti_split,
    int((ti.max() - ti.min()) / (0.08)),
    histtype="bar",
    stacked=True,
    color=colors,
    alpha=alpha,
)
# lower_axes[2].set_title("TI")
lower_axes[2].text(
    *id_pos,
    figure_ids[6],
    horizontalalignment="center",
    verticalalignment="center",
    transform=lower_axes[2].transAxes,
)
lower_axes[2].set_xlabel(r"$I_0$ [-]")
lower_axes[2].set_ylabel(r"Number of samples [-]")


# for u_, ti_, color, marker_type in zip(u_split, ti_split, colors, markers):
#     lower_axes[3].scatter(u_, ti_, color=color, alpha=alpha, marker=marker_type)

combined_u = np.concatenate(u_split)
combined_ti = np.concatenate(ti_split)
labels_u = np.concatenate(
    [np.full(len(s), i) for i, s in enumerate(u_split)]
)  # 0=train,1=val,2=test

rng = np.random.default_rng(0)  # fixed seed for reproducibility (remove seed if undesired)
perm = rng.permutation(len(labels_u))
# drop 80 percent for plotting speed
perm = perm[::downsampling_factor]


u_shuf = combined_u[perm]
ti_shuf = combined_ti[perm]
labels_shuf = labels_u[perm]

chunk_size = 2000  # adjust: smaller -> more interleaving but slower
for start in range(0, len(labels_shuf), chunk_size):
    end = start + chunk_size
    for cls in range(len(colors)):
        mask = labels_shuf[start:end] == cls

        if mask.any():
            lower_axes[3].scatter(
                u_shuf[start:end][mask],
                ti_shuf[start:end][mask],
                color=colors[cls],
                marker=markers[cls],
                alpha=alpha,
                s=10,
                edgecolor="k",
                linewidths=0.5,
            )


# plot limits of u vs ti
lower_axes[3].plot(u_lin, TI_upper, color="black", linestyle="-", linewidth=1)
lower_axes[3].plot(u_lin, TI_lower, color="black", linestyle="-", linewidth=1)
lower_axes[3].plot(
    [cutin, cutin],
    [TI_lower[0], TI_upper[0]],
    color="black",
    linestyle="-",
    linewidth=1,
)
lower_axes[3].plot(
    [cutout, cutout],
    [TI_lower[-1], TI_upper[-1]],
    color="black",
    linestyle="-",
    linewidth=1,
)


# lower_axes[3].set_title("Wind speed and TI")
lower_axes[3].text(
    *id_pos,
    figure_ids[7],
    horizontalalignment="center",
    verticalalignment="center",
    transform=lower_axes[3].transAxes,
)
lower_axes[3].set_xlabel(r"$U\ [\mathrm{ms}^{-1}]$")
lower_axes[3].set_ylabel(r"$I_0$ [-]")

marker_size = 10
legend_elements = [
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
    Line2D(
        [0],
        [0],
        marker=markers[0],
        color="w",
        label="Training data",
        markerfacecolor=colors[0],
        markersize=marker_size,
    ),
    Line2D(
        [0],
        [0],
        marker=markers[1],
        color="w",
        label="Validation data",
        markerfacecolor=colors[1],
        markersize=marker_size,
    ),
    Line2D(
        [0],
        [0],
        marker=markers[2],
        color="w",
        label="Test data",
        markerfacecolor=colors[2],
        markersize=marker_size,
    ),
    Line2D(
        [0],
        [0],
        color="k",
        label="Distribution bondary",
        lw=1,
    ),
]

figure.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=5,
    borderaxespad=0.0,
)


plt.tight_layout()
plt.savefig(_THIS_DIR / "figures" / "layout_and_inflow.pdf", bbox_inches="tight")
# plt.savefig(_THIS_DIR / "figures" / "layout_and_inflow.png", bbox_inches="tight", dpi=300)
