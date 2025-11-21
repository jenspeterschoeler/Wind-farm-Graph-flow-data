import logging
import os
import tempfile
from zipfile import ZipFile

import numpy as np
import torch
from joblib import Parallel, delayed
from py_wake import HorizontalGrid
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.examples.data.dtu10mw import power_curve as power_curve_dtu10mw
from tqdm import tqdm

from inflow_generator import IEC_61400_1_2019_class_interpreter, InflowGenerator
from layout_generator import (
    PLayGen,
    sample_truncated_normal_floats,
    sample_truncated_normal_integers,
)
from pre_process import pre_process
from run_pywake import simulate_farm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Initiating script")


num_cpu = int(os.cpu_count() / 2)
dataset_path = os.path.abspath("./data/large_graphs_nodes_2_v2")
os.makedirs(dataset_path, exist_ok=True)


target_samples = 5000  # will be multiplied by inflows_per_layout (10)
extra_samples = int(target_samples * 0.1)
total_samples = target_samples + extra_samples

n_wts_all = sample_truncated_normal_integers(
    low=20, high=100, mean=60, std=60, size=total_samples
)
wt_spacings_all = sample_truncated_normal_floats(
    low=2, high=8, mean=5, std=3, size=total_samples
)
types_all = np.random.choice(
    ["cluster", "single string", "multiple string", "parallel string"],
    total_samples,
    p=[0.4, 0.2, 0.2, 0.2],
    replace=True,
)


# n_wts_all = sample_truncated_normal_integers(
#     low=40, high=120, mean=80, std=50, size=total_samples
# )
# wt_spacings_all = sample_truncated_normal_floats(
#     low=2, high=10, mean=5, std=3, size=total_samples
# )
types_all = np.random.choice(
    ["cluster", "single string", "multiple string", "parallel string"],
    total_samples,
    p=[0.4, 0.2, 0.2, 0.2],
    replace=True,
)

n_wts = n_wts_all[:target_samples]
wt_spacings = wt_spacings_all[:target_samples]
types = types_all[:target_samples]

n_wts_extra = n_wts_all[target_samples:]
wt_spacings_extra = wt_spacings_all[target_samples:]
types_extra = types_all[target_samples:]

## Generate layouts
layout_generator = PLayGen(
    D=1,
)

# ## Generate random wind farm layout
all_layouts = []
layout_types_used = []
wt_spacings_used = []
failed_layouts = 0

for n_wt, wt_spacing, type_ in zip(n_wts, wt_spacings, types):
    layout_generator.set_N_turbs(n_wt)
    layout_generator.set_spacing(wt_spacing)
    layout_generator.set_layout_style(type_)
    try:
        layout = layout_generator()
    except Exception as e:
        print(e)
        failed_layouts += 1
        continue
    all_layouts.append(layout)
    layout_types_used.append(type_)
    wt_spacings_used.append(wt_spacing)

for n_wt, wt_spacing, type_ in zip(n_wts_extra, wt_spacings_extra, types_extra):
    layout_generator.set_N_turbs(n_wt)
    layout_generator.set_spacing(wt_spacing)
    layout_generator.set_layout_style(type_)
    try:
        layout = layout_generator()
    except Exception as e:
        print(e)
        continue

    all_layouts.append(layout)
    layout_types_used.append(type_)
    wt_spacings_used.append(wt_spacing)
    if len(all_layouts) >= target_samples:
        break


x_min_layout = 1e9
y_min_layout = 1e9
x_max_layout = -1e9
y_max_layout = -1e9
for layout in all_layouts:
    x_min_layout = np.min([layout[:, 0].min(), x_min_layout])
    y_min_layout = np.min([layout[:, 1].min(), y_min_layout])
    x_max_layout = np.max([layout[:, 0].max(), x_max_layout])
    y_max_layout = np.max([layout[:, 1].max(), y_max_layout])


x_min = x_min_layout - 2
x_max = x_max_layout + 100
x_range = x_max - x_min
y_min = y_min_layout
y_max = y_max_layout
y_range = y_max - y_min


# Turbine settings
wt = DTU10MW()
D = wt.diameter()
cut_in = power_curve_dtu10mw[:, 0].min()
cut_out = power_curve_dtu10mw[:, 0].max()
turbine_settings = {
    "cutin_u": cut_in,  # cut-in wind speed [m/s]
    "cutout_u": cut_out,  # cut-out wind speed [m/s]
    "height_above_ground": wt.hub_height(),  # hub height [m]
}


# grid_density = 2  # points per D.
# x_grid = np.linspace(x_min * D, x_max * D, int(x_range * grid_density))
# y_grid = np.linspace(y_min * D, y_max * D, int(y_range * grid_density))
# grid = HorizontalGrid(x=x_grid, y=y_grid)
# X, Y = np.meshgrid(x_grid, y_grid)

# # Construct weighting map
# probabilities_combined = combined_weighting(
#     X, Y, x_min, step_pos=x_max_layout * D, clip_min=0.3
# )

# org_shape = np.array(X.shape)
# np.savez(
#     os.path.join(dataset_path, "probabilities_combined"),
#     probabilities=probabilities_combined,
#     org_shape=org_shape,
# )

## Calculate flow conditions
# Inflow settings
inflow_settings = IEC_61400_1_2019_class_interpreter("I", "B")

# Generate inflow generator
inflow_gen = InflowGenerator(
    inflow_settings=inflow_settings, turbine_settings=turbine_settings
)

# Generate inflows
inflows_per_layout = 10
n_samples_inflow = inflows_per_layout * len(all_layouts)
inflows = inflow_gen.generate_inflows(n_samples_inflow, output_type="array")
layout_inflows_list = np.split(inflows, len(all_layouts))


inflow_dicts = []
layout_metrics = []
grids = []
idxs_buffer = []
layout_stats_dicts = []

to_graph_kws = {
    "connectivity": "delaunay",
    "add_edge": "cartesian",
    "rel_wd": None,
}

## Run pywake
for i, (layout, layout_type, wt_spacing, inflows) in tqdm(
    enumerate(
        zip(all_layouts, layout_types_used, wt_spacings_used, layout_inflows_list)
    )
):

    u = inflows[:, 0]
    ti = inflows[:, 1]
    inflow_dict = {
        "u": u,
        "ti": ti,
    }

    layout_stats_dict = {
        "layout_stats": {
            "layout_type": layout_type,
            "wt_spacing": wt_spacing,
            "n_wt": layout.shape[0],
        }
    }

    x_min_layout = layout[:, 0].min()
    y_min_layout = layout[:, 1].min()
    x_max_layout = layout[:, 0].max()
    y_max_layout = layout[:, 1].max()

    x_min = x_min_layout - 10
    x_max = x_max_layout + 100
    x_range = x_max - x_min
    y_min = y_min_layout - 5
    y_max = y_max_layout + 5
    y_range = y_max - y_min

    grid_density = 3  # points per D.
    x_grid = np.linspace(x_min * D, x_max * D, int(x_range * grid_density))
    y_grid = np.linspace(y_min * D, y_max * D, int(y_range * grid_density))
    grid = HorizontalGrid(x=x_grid, y=y_grid)

    layout_metric = layout * D
    inflow_dicts.append(inflow_dict)
    layout_metrics.append(layout_metric)
    grids.append(grid)
    layout_stats_dicts.append(layout_stats_dict)
    convert_to_graph = True
    idxs_buffer.append(i)
    # Run pywake every time we have num_cpu layouts
    if (i + 1) % (num_cpu) == 0 or i == len(all_layouts):
        outputs = Parallel(n_jobs=num_cpu)(
            delayed(simulate_farm)(
                inflow_dict,
                layout_metric,
                grid,
                convert_to_graph,
                dict(to_graph_kws, **layout_stats_dict),
            )
            for inflow_dict, layout_metric, grid, layout_stats_dict in zip(
                inflow_dicts, layout_metrics, grids, layout_stats_dicts
            )
        )

        if convert_to_graph:
            # Save contetns of outputs to files
            with tempfile.TemporaryDirectory() as tempdir:
                for layout_num, output_tuple in zip(idxs_buffer, outputs):
                    output, original_trunk_shape = output_tuple
                    zip_path = (
                        os.path.join(dataset_path, f"_layout{layout_num}") + ".zip"
                    )
                    with ZipFile(zip_path, "w") as zf:
                        for graph in output:

                            graph_file_name = f"_layout{layout_num}_ws{str(np.round(graph.global_features[0].item(), 2))}_ti_{graph.global_features[1].item()}.pt"

                            graph_temp_path = os.path.join(tempdir, graph_file_name)
                            torch.save(graph, graph_temp_path)

                            zf.write(filename=graph_temp_path, arcname=graph_file_name)

        # Reset lists
        idxs_buffer = []
        inflow_dicts = []
        layout_metrics = []
        grids = []
        layout_stats_dicts = []

logger.info("Scaling and splitting data")
total_desired_samples = target_samples * inflows_per_layout


pre_process(
    dataset_path,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    scaling_method="run4",
    original_trunk_shape=original_trunk_shape,
)

logger.info("Done")
