[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17671257.svg)](https://doi.org/10.5281/zenodo.17671257)

# GNO Dataset Generation

Generate graph-structured wind farm datasets using PyWake simulations. Outputs are formatted for PyTorch Geometric (PyG), enabling training with Graph Neural Operators (GNO) for wind farm flow prediction.

> ![plot](./experiments/layout_and_inflow.png)
> Procedurally generated data: (a-d) wind farm layouts using PLayGen, (e) Quasi-random samples via Sobol sequence, (f) Wind speed distribution, (g) TI distribution, (h) Generated U and TI with boundary.

## Quick Start

```bash
# Install environment
pixi install

# Run test dataset generation (10 layouts, ~5 min)
pixi run python main.py --config turbopark10_test

# Run full dataset generation (250 layouts, ~hours)
pixi run python main.py --config turbopark250
```

## Project Structure

```
data-generation/
├── main.py                    # Main dataset generation pipeline
├── run_pywake.py              # PyWake simulation wrapper
├── to_graph.py                # Graph construction utilities
├── layout_generator.py        # Wind farm layout generation (PLayGen)
├── inflow_generator.py        # Inflow condition sampling (IEC 61400-1)
├── pre_process.py             # Dataset preprocessing and splitting
├── convert_awf_to_graphs.py   # AWF database conversion + preprocessing
├── load_shared_layouts.py     # Load shared layout/inflow metadata
├── submit_turbopark_250layouts.sh  # SLURM submission script
├── experiments/               # Visualization and inspection scripts
├── utils/                     # Utility functions
├── archive/                   # Archived documentation
└── data/                      # Generated datasets (gitignored)
```

## Dataset Generation

### PyWake Datasets (main.py)

Generate datasets with configurable wake models:

```bash
# Test configuration (10 layouts, 4 inflows each)
pixi run python main.py --config turbopark10_test

# Full configuration (250 layouts, 20 inflows each)
pixi run python main.py --config turbopark250
```

**Available configurations:**
- `turbopark10_test`: Quick test (10 layouts, grid_density=1)
- `turbopark250`: Full dataset (250 layouts, grid_density=3)

**Output structure:**
```
data/turbopark_250layouts/
├── _layout0.zip          # Graph data per layout
├── _layout1.zip
├── ...
├── layouts_metadata.npz  # Layout configurations (tracked in git)
├── inflows_metadata.npz  # Inflow conditions (tracked in git)
├── stats.json            # Dataset statistics
└── scale_stats.json      # Scaling parameters
```

### AWF Database Conversion

Convert the AWF (Aventa Wind Farm) database to GNO-compatible format. The script handles both conversion and preprocessing in one step:

```bash
# Convert and preprocess AWF database (10 layouts for testing)
pixi run python convert_awf_to_graphs.py \
    --database data/awf_database.nc \
    --output data/awf_graphs_test \
    --max-layouts 10

# Full conversion (all layouts)
pixi run python convert_awf_to_graphs.py \
    --database data/awf_database.nc \
    --output data/awf_graphs

# Conversion only (skip preprocessing)
pixi run python convert_awf_to_graphs.py \
    --database data/awf_database.nc \
    --output data/awf_graphs \
    --skip-preprocessing
```

**Options:**
- `--max-layouts N`: Limit to first N layouts (for testing)
- `--train-size 0.6`: Training set fraction (default: 0.6)
- `--val-size 0.2`: Validation set fraction (default: 0.2)
- `--test-size 0.2`: Test set fraction (default: 0.2)
- `--skip-preprocessing`: Only convert, don't preprocess

**Note:** AWF coordinates are in rotor diameters. The converter handles unit conversion to meters automatically. Domain clipping is applied to match the PyWake grid format.

## Wake Models

The `run_pywake.py` module supports configurable wake models:

| Model | Description | Use Case |
|-------|-------------|----------|
| `TurboGaussianDeficit` | TurbOPark Gaussian wake | Default for main.py |
| `NiayifarGaussianDeficit` | Gaussian with turbulence | Alternative |

```python
from run_pywake import create_wake_config
from py_wake.deficit_models import NOJDeficit

config = create_wake_config(
    deficit_model=NOJDeficit(),
    wind_farm_model="PropagateDownwind"
)
```

## Grid Generation

Grids are layout-dependent with configurable bounds:

```python
from run_pywake import create_grid_for_layout

grid = create_grid_for_layout(
    layout,              # Turbine positions [N, 2] in diameters
    turbine_diameter=D,  # Turbine diameter in meters
    grid_density=3,      # Grid points per diameter
    x_upstream=10.0,     # Upstream extent in D
    x_downstream=100.0,  # Downstream extent in D
    y_margin=5.0,        # Cross-stream margin in D
)
```

## Graph Format

Each graph contains:
- `pos`: Turbine positions [N_wt, 2] in meters
- `node_features`: Turbine wind speeds [N_wt, 1]
- `edge_index`: Graph connectivity
- `edge_attr`: Edge features (distances)
- `trunk_inputs`: Probe positions [N_probes, 2]
- `output_features`: Flow velocities at probes [N_probes, 1]
- `global_features`: [freestream_ws, turbulence_intensity]

## SLURM Submission

For HPC clusters:

```bash
sbatch submit_turbopark_250layouts.sh
```

## Inspection Tools

Visualize generated datasets:

```bash
# Inspect TurbOPark dataset
pixi run python experiments/inspect_turbopark_dataset.py \
    --data-dir data/turbopark_10layouts_test

# Inspect AWF dataset
pixi run python experiments/inspect_awf_dataset.py \
    --data-dir data/awf_graphs
```

## Data Sharing

Generated datasets are gitignored, but metadata files are tracked:
- `layouts_metadata.npz` - Layout configurations for reproduction
- `inflows_metadata.npz` - Inflow conditions for reproduction

Collaborators can regenerate identical datasets using:

```python
from load_shared_layouts import load_layouts_and_inflows

layouts, metadata, inflows = load_layouts_and_inflows("data/turbopark_250layouts")
```

## References

- **PLayGen** (Layout Generation): https://github.com/NREL/WPGNN
- **IEC 61400-1** (Inflow Generation): [DOI: 10.5194/wes-3-767-2018](https://doi.org/10.5194/wes-3-767-2018)
- **PyWake**: https://topfarm.pages.windenergy.dtu.dk/PyWake/
