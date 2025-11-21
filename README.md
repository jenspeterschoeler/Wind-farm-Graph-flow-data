[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17671257.svg)](https://doi.org/10.5281/zenodo.17671257)

# Graph Dataset Generation

Utilities for generating graph-structured datasets using PyWake wind-farm simulations. Outputs are formatted for PyTorch Geometric (PyG), enabling training and experimentation with Graph Neural Networks (GNNs) for wind-farm flow modeling.

> ![plot](./experiments/layout_and_inflow.png)
 Procedurally generated data, (a-d) wind farm layouts using PLayGen. (e) Quasi random samples generated with the Sobol sequence.
(f) U distribution. (g) TI distribution. (h) Generated U and TI with boundary

🚀 Getting Started

To run the included environment clone the repo, install the environment and run `main.py`.

1. Install Pixi

    Follow instructions for your platform:
    👉 https://pixi.sh/

2. Set Up the Environment 
    From the project root (where pixi.toml is located):
    ```bash
    pixi install
    ```
    Run script:
    ```bash
    pixi run python main.py
    ```
    *Alternative as a shell*
    ```bash
    pixi shell
    python main.py
    ```

🔧 Layout & Inflow Generation

PlayGen — Generates the wind-farm layouts, defining turbine positions for the dataset. PlayGen supports configurable spacing, count, and site geometry.

↳ GitHub repo: https://github.com/NREL/WPGNN

Winds-to-Loads — Produces realistic atmospheric inflows (turbulence, wind speed, direction) that feed into PyWake, providing accurate node features and labels for GNN training.

↳ DOI: [10.5194/wes‑3‑767‑2018](https://orbit.dtu.dk/en/publications/from-wind-to-loads-wind-turbine-site-specific-load-estimation-wit/?utm_source=chatgpt.com)

