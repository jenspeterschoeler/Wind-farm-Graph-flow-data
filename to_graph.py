import math
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.transforms import Cartesian, Delaunay, Distance, FaceToEdge, Polar


class PyGTupleData(PyGData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "global_features":
            return None
        if key == "n_node":
            return None
        if key == "n_edge":
            return None
        if key == "trunk_inputs":
            return None
        if key == "output_features":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def to_graph(
    points: np.ndarray,
    connectivity: str = "delaunay",
    add_edge: str = "polar",
    node_features: np.ndarray = None,
    global_features: np.ndarray = None,
    rel_wd: float = 270,
    trunk_inputs=None,
    output_features=None,
    layout_stats: Dict = None,
) -> PyGTupleData:
    """
    Converts np.array to torch_geometric.data.data.Data object with the specified connectivity and edge feature type.
    """
    assert connectivity in [
        "delaunay",
    ]
    assert points.shape[1] == 2

    points_ = torch.Tensor(points)
    raw_graph_data = PyGTupleData(pos=points_)
    if connectivity.casefold() == "delaunay":
        delaunay_fn = Delaunay()
        edge_fn = FaceToEdge()
        graph = edge_fn(delaunay_fn(raw_graph_data))

    else:
        raise ValueError(
            "Please define the connectivity scheme (available types: : 'delaunay')"
        )

    if add_edge == "polar".casefold():
        polar_fn = Polar(norm=False)
        graph = polar_fn(graph)
        if rel_wd is not None:
            edge_rel_wd = math.radians(270) - graph.edge_attr[:, 1]
            graph.edge_attr = torch.cat(
                (graph.edge_attr, edge_rel_wd.unsqueeze(1)), dim=1
            )
    elif add_edge == "cartesian".casefold():
        cartesian_fn = Cartesian(norm=False)
        distance_fn = Distance(norm=False)
        graph = cartesian_fn(graph)
        graph = distance_fn(graph)

    else:
        raise ValueError(
            "Please select a coordinate system that is supported (available types: : 'polar')"
        )

    graph.n_node = torch.Tensor([node_features.shape[0]])
    graph.n_edge = torch.Tensor([graph.edge_attr.shape[0]])

    if node_features is not None:
        graph.node_features = torch.Tensor(node_features)

    if global_features is not None:
        graph.global_features = torch.Tensor(global_features)

    if trunk_inputs is not None:
        graph.trunk_inputs = torch.Tensor(trunk_inputs)

    if output_features is not None:
        graph.output_features = torch.Tensor(output_features)

    if layout_stats is not None:
        for k, v in layout_stats.items():
            setattr(graph, k, v)

    graph = PyGTupleData(
        **graph
    )  # Convert at the end in case altering __cat_dim__ breaks something
    return graph


if __name__ == "__main__":
    points = np.array([[0, 0], [1, 0], [0, 1], [2, 2]])
    node_features = np.array([[10], [11], [12], [13]])
    global_features = np.array([1, 2])
    trunk_inputs = np.ones((12, 12, 2))
    output_features = np.ones((12, 12, 1))
    graph = to_graph(
        points,
        node_features=node_features,
        global_features=global_features,
        trunk_inputs=trunk_inputs,
        output_features=output_features,
        rel_wd=270,
    )

    print(graph)

    graph = to_graph(
        points,
        node_features=node_features,
        global_features=global_features,
        trunk_inputs=trunk_inputs,
        output_features=output_features,
        rel_wd=None,
        add_edge="cartesian",
    )
    print(graph)

    # print(graph.edge_index)
    # print(graph.edge_attr)
    # print(graph.pos)
    # print(graph.node_features)
    # print(graph.global_features)
    # print(graph.trunk_inputs)
    # print(graph.output_features)
