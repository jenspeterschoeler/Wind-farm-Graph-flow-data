from typing import Dict

import numpy as np
import pandas as pd
from py_wake import HorizontalGrid
from py_wake.deficit_models import NiayifarGaussianDeficit, SelfSimilarityDeficit2020
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.turbulence_models import CrespoHernandez

# LinearSum
from py_wake.wind_farm_models import All2AllIterative, PropagateDownwind

from to_graph import to_graph


def simulate_farm(
    inflow_dict: Dict,
    positions: np.ndarray,
    grid: HorizontalGrid,
    convert_to_graph=False,
    to_graph_kws: Dict = None,
):
    """Function to simulate the power and loads of a wind farm given the inflow conditions and the
    wind turbine positions using PyWake. The function will return the simulated power and loads
    for each turbine.

    args:
    inflow_df: pd.DataFrame, the inflow conditions for the wind farm
    positions: np.ndarray, the wind turbine positions
    loads_method: str, the kind of load model to use: either OneWT or TwoWT
    """
    site = UniformSite()

    ws = inflow_dict["u"]
    wd = np.ones(len(ws)) * 270
    ti = inflow_dict["ti"]
    x = positions[:, 0]
    y = positions[:, 1]

    wt = DTU10MW()

    # wf_model = PropagateDownwind(
    #     site,
    #     wt,
    #     wake_deficitModel=NiayifarGaussianDeficit(),
    #     superpositionModel=LinearSum(),
    #     turbulenceModel=CrespoHernandez(),
    # )

    wf_model = All2AllIterative(
        site,
        wt,
        wake_deficitModel=NiayifarGaussianDeficit(use_effective_ws=True),
        blockage_deficitModel=SelfSimilarityDeficit2020(),
        superpositionModel=LinearSum(),
        turbulenceModel=CrespoHernandez(),
    )

    farm_sims = []
    flow_maps = []
    for ws_, wd_, ti_ in zip(ws, wd, ti):
        farm_sim = wf_model(
            x,
            y,  # wind turbine positions
            wd=wd_,  # Wind direction
            ws=ws_,  # Wind speed
            TI=ti_,  # Turbulence intensity
        )

        flow_map = farm_sim.flow_map(grid=grid, wd=wd_, ws=ws_)
        farm_sims.append(farm_sim)
        flow_maps.append(flow_map)

    xx, yy = np.meshgrid(grid.x, grid.y)
    x_grid = xx.flatten()
    y_grid = yy.flatten()
    xy_grid = np.array([x_grid, y_grid]).T

    farm_sim["nwt_final"] = len(x)
    if convert_to_graph:
        graph_list = []
        for farm_sim, flowmap in zip(farm_sims, flow_maps):
            wt_ws = farm_sim.WS_eff.values.copy().squeeze()
            # TODO Add these as optionals e.g. extra stuff in the graph, consider backwards compatability?
            wt_ti = farm_sim.TI_eff.values.copy().squeeze()
            wt_CTs = farm_sim.CT.values.copy().squeeze()
            wts_info = np.array([wt_ws, wt_ti, wt_CTs]).T
            u_output = flowmap.WS_eff.values.copy().squeeze()
            original_trunk_shape = u_output.shape
            u_output = u_output.flatten()
            output_features = u_output.reshape(-1, 1)
            global_features = np.array([farm_sim["WS"], farm_sim["TI"]]).copy()
            node_features = wts_info  # wt_CTs.reshape(-1, 1)
            points = np.array([farm_sim["x"], farm_sim["y"]]).T.copy()
            graph = to_graph(
                points=points,
                node_features=node_features,
                global_features=global_features,
                trunk_inputs=xy_grid,
                output_features=output_features,
                **to_graph_kws,
            )
            graph_list.append(graph)
        return graph_list, original_trunk_shape
    else:
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
    flow_maps, farm_sims = simulate_farm(
        inflow_dict={
            "u": np.array([10.0, 12, 14, 16, 18, 20]),
            "ti": np.array([0.1, 0.1, 0.1, 0.1, 0.2]),
        },
        positions=wt_positions,
        grid=grid,
    )

    for flow_map in flow_maps:
        x, y = flow_map.x, flow_map.y
        flow_map.WS_eff.squeeze().plot.contourf(x="x", y="y")
        plt.scatter(wt_positions[:, 0], wt_positions[:, 1], c="k")
        plt.show()

    to_graph_kws = {
        "connectivity": "delaunay",
        "add_edge": "cartesian",
        "rel_wd": None,
    }

    graphs = simulate_farm(
        inflow_dict={
            "u": np.array([10, 12, 14, 16, 18]),
            "ti": np.array([0.1, 0.1, 0.1, 0.1]),
        },
        positions=wt_positions,
        grid=grid,
        convert_to_graph=True,
        to_graph_kws=to_graph_kws,
    )

    graph = graphs[0]
