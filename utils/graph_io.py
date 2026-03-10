"""
Graph I/O utilities for saving and loading PyTorch Geometric graphs.

This module consolidates duplicated graph saving code across the codebase.
"""

import os
import tempfile
from zipfile import ZipFile

import numpy as np
import torch

# Standard filename template for graphs
GRAPH_FILENAME_TEMPLATE = "_layout{layout_num}_ws{ws_val}_ti_{ti_val}.pt"


def save_graphs_to_zip(graphs: list, layout_num: int, output_dir: str) -> str:
    """
    Save a list of graphs to a zip file with standard naming convention.

    Args:
        graphs: List of PyTorch Geometric Data objects
        layout_num: Layout index number
        output_dir: Directory to save the zip file

    Returns:
        str: Path to the created zip file

    The graphs are saved with filenames following the pattern:
        _layout{N}_ws{ws}_ti_{ti}.pt

    Where ws and ti are extracted from graph.global_features[0] and [1].
    """
    zip_path = os.path.join(output_dir, f"_layout{layout_num}.zip")

    with tempfile.TemporaryDirectory() as tempdir, ZipFile(zip_path, "w") as zf:
        for graph in graphs:
            ws_val = np.round(graph.global_features[0].item(), 2)
            ti_val = graph.global_features[1].item()
            filename = f"_layout{layout_num}_ws{ws_val}_ti_{ti_val}.pt"

            temp_path = os.path.join(tempdir, filename)
            torch.save(graph, temp_path)
            zf.write(filename=temp_path, arcname=filename)

    return zip_path
