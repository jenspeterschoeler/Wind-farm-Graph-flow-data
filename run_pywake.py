"""PyWake wind farm simulation runner."""

import gc

import numpy as np
from py_wake import HorizontalGrid
from py_wake.deficit_models import NiayifarGaussianDeficit, SelfSimilarityDeficit2020
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.superposition_models import LinearSum
from py_wake.turbulence_models import CrespoHernandez

from to_graph import to_graph
from utils.pywake_utils import DEFAULT_TO_GRAPH_KWS, create_wind_farm_model_fresh


def create_wake_config(
    deficit_model=None,
    blockage_model=None,
    superposition_model=None,
    turbulence_model=None,
    wind_farm_model="All2AllIterative",
):
    """
    Create a wake configuration dictionary for PyWake simulations.

    Args:
        deficit_model: Wake deficit model instance (default: NiayifarGaussianDeficit)
        blockage_model: Blockage model instance (default: SelfSimilarityDeficit2020)
        superposition_model: Superposition model instance (default: LinearSum)
        turbulence_model: Turbulence model instance (default: CrespoHernandez)
        wind_farm_model: 'All2AllIterative' or 'PropagateDownwind' (default: 'All2AllIterative')

    Returns:
        dict: Wake configuration dictionary for use with simulate_farm()

    Examples:
        # Default configuration (Niayifar + blockage)
        config = create_wake_config()

        # Custom wake model
        from py_wake.deficit_models import NOJDeficit
        config = create_wake_config(deficit_model=NOJDeficit())

        # Downwind propagation only (no blockage)
        config = create_wake_config(wind_farm_model="PropagateDownwind")
    """
    config = {}

    if deficit_model is not None:
        config["deficit_model"] = deficit_model
    else:
        config["deficit_model"] = NiayifarGaussianDeficit(use_effective_ws=True)

    if blockage_model is not None:
        config["blockage_model"] = blockage_model
    elif wind_farm_model == "All2AllIterative":
        # Only add blockage model for All2AllIterative
        config["blockage_model"] = SelfSimilarityDeficit2020()

    if superposition_model is not None:
        config["superposition_model"] = superposition_model
    else:
        config["superposition_model"] = LinearSum()

    if turbulence_model is not None:
        config["turbulence_model"] = turbulence_model
    else:
        config["turbulence_model"] = CrespoHernandez()

    config["wind_farm_model"] = wind_farm_model

    return config


def create_grid_for_layout(
    layout: np.ndarray,
    turbine_diameter: float,
    grid_density: int = 3,
    x_upstream: float = 10.0,
    x_downstream: float = 100.0,
    y_margin: float = 5.0,
):
    """
    Create a HorizontalGrid for a given wind farm layout.

    Args:
        layout: Turbine positions [N, 2] array in diameters (not meters)
        turbine_diameter: Turbine diameter in meters
        grid_density: Grid points per diameter (default: 3)
        x_upstream: Upstream extent in diameters (default: 10)
        x_downstream: Downstream extent in diameters (default: 100)
        y_margin: Lateral margin in diameters (default: 5)

    Returns:
        HorizontalGrid: PyWake grid object

    Example:
        from py_wake.examples.data.dtu10mw import DTU10MW
        wt = DTU10MW()
        layout = np.array([[0, 0], [5, 0], [10, 0]])  # in diameters
        grid = create_grid_for_layout(layout, wt.diameter())
    """
    x_min_layout = layout[:, 0].min()
    y_min_layout = layout[:, 1].min()
    x_max_layout = layout[:, 0].max()
    y_max_layout = layout[:, 1].max()

    x_min = x_min_layout - x_upstream
    x_max = x_max_layout + x_downstream
    x_range = x_max - x_min
    y_min = y_min_layout - y_margin
    y_max = y_max_layout + y_margin
    y_range = y_max - y_min

    x_grid = np.linspace(
        x_min * turbine_diameter, x_max * turbine_diameter, int(x_range * grid_density)
    )
    y_grid = np.linspace(
        y_min * turbine_diameter, y_max * turbine_diameter, int(y_range * grid_density)
    )

    return HorizontalGrid(x=x_grid, y=y_grid)


def simulate_farm(
    inflow_dict: dict,
    positions: np.ndarray,
    grid: HorizontalGrid,
    convert_to_graph=False,
    to_graph_kws: dict = None,
    wake_config: dict = None,
):
    """
    Simulate wind farm flow using PyWake with configurable wake models.

    Args:
        inflow_dict: Dictionary with 'u' (wind speeds) and 'ti' (turbulence intensities)
        positions: Turbine positions [N, 2] array in meters
        grid: PyWake HorizontalGrid for flow field computation
        convert_to_graph: If True, convert outputs to graph format
        to_graph_kws: Keyword arguments for to_graph() function
        wake_config: Optional dict to configure wake models with keys:
            - 'deficit_model': Wake deficit model instance (default: NiayifarGaussianDeficit)
            - 'blockage_model': Blockage model instance (default: SelfSimilarityDeficit2020)
            - 'superposition_model': Superposition model instance (default: LinearSum)
            - 'turbulence_model': Turbulence model instance (default: CrespoHernandez)
            - 'wind_farm_model': 'All2AllIterative' or 'PropagateDownwind' (default: 'All2AllIterative')

    Returns:
        If convert_to_graph=True: (list of graphs, original_trunk_shape)
        If convert_to_graph=False: (flow_maps, farm_sims)
    """
    ws = inflow_dict["u"]
    wd = np.ones(len(ws)) * 270
    ti = inflow_dict["ti"]
    x = positions[:, 0]
    y = positions[:, 1]

    # Precompute grid coordinates once
    xx, yy = np.meshgrid(grid.x, grid.y)
    x_grid = xx.flatten()
    y_grid = yy.flatten()
    xy_grid = np.array([x_grid, y_grid]).T
    del xx, yy, x_grid, y_grid  # Free intermediate arrays

    if convert_to_graph:
        graph_list = []
        original_trunk_shape = None

        # Process each inflow one at a time to minimize memory
        for i, (ws_, wd_, ti_) in enumerate(zip(ws, wd, ti)):
            # Create fresh model instances for each inflow to prevent state accumulation
            wf_model, site, wt = create_wind_farm_model_fresh(wake_config)

            farm_sim = wf_model(x, y, wd=wd_, ws=ws_, TI=ti_)
            flow_map = farm_sim.flow_map(grid=grid, wd=wd_, ws=ws_)

            # Extract data immediately and delete xarray objects
            wt_ws = farm_sim.WS_eff.values.copy().squeeze()
            u_output = flow_map.WS_eff.values.copy().squeeze()
            if original_trunk_shape is None:
                original_trunk_shape = u_output.shape
            global_features = np.array([float(farm_sim["WS"]), float(farm_sim["TI"])])
            points = np.array([farm_sim["x"].values, farm_sim["y"].values]).T.copy()

            # Delete xarray and model objects immediately after extracting data
            del farm_sim, flow_map, wf_model, site, wt

            # Build graph
            output_features = u_output.flatten().reshape(-1, 1)
            node_features = wt_ws.reshape(-1, 1)
            del u_output, wt_ws  # Free intermediate arrays

            graph = to_graph(
                points=points,
                node_features=node_features,
                global_features=global_features,
                trunk_inputs=xy_grid,
                output_features=output_features,
                **to_graph_kws,
            )
            graph_list.append(graph)
            del points, node_features, global_features, output_features

            # Periodic garbage collection every 5 inflows
            if (i + 1) % 5 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()

        return graph_list, original_trunk_shape
    else:
        # Non-graph mode: accumulate all results (for visualization/debugging)
        wf_model, site, wt = create_wind_farm_model_fresh(wake_config)

        farm_sims = []
        flow_maps = []
        for ws_, wd_, ti_ in zip(ws, wd, ti):
            farm_sim = wf_model(x, y, wd=wd_, ws=ws_, TI=ti_)
            flow_map = farm_sim.flow_map(grid=grid, wd=wd_, ws=ws_)
            farm_sims.append(farm_sim)
            flow_maps.append(flow_map)
        return flow_maps, farm_sims


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    wt = DTU10MW()
    D = wt.diameter()
    x = np.linspace(7, 50, 50) * D
    y = np.linspace(-10, 10, 25) * D

    grid = HorizontalGrid(
        x=x,
        y=y,
    )
    wt_positions = np.array([[0, 0], [5, 3], [0, 5], [5, 5]]) * D

    # Demo: non-graph mode (visualization)
    flow_maps, farm_sims = simulate_farm(
        inflow_dict={
            "u": np.array([10.0, 12.0, 14.0, 16.0]),
            "ti": np.array([0.1, 0.1, 0.15, 0.2]),
        },
        positions=wt_positions,
        grid=grid,
    )

    for flow_map in flow_maps:
        x, y = flow_map.x, flow_map.y
        flow_map.WS_eff.squeeze().plot.contourf(x="x", y="y")
        plt.scatter(wt_positions[:, 0], wt_positions[:, 1], c="k")
        plt.show()

    # Demo: graph mode (using DEFAULT_TO_GRAPH_KWS from utils)
    graphs, trunk_shape = simulate_farm(
        inflow_dict={
            "u": np.array([10.0, 12.0, 14.0, 16.0]),
            "ti": np.array([0.1, 0.1, 0.15, 0.2]),
        },
        positions=wt_positions,
        grid=grid,
        convert_to_graph=True,
        to_graph_kws=DEFAULT_TO_GRAPH_KWS,
    )

    print(f"Generated {len(graphs)} graphs with trunk shape {trunk_shape}")
    graph = graphs[0]
