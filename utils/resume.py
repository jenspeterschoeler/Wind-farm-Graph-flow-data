"""
Utility functions for resuming dataset generation.

Provides helpers to detect completed layouts and load saved layouts/inflows
from NPZ files, enabling interrupted generation runs to continue from where
they left off.
"""

import os

import numpy as np


def get_completed_layouts(output_dir: str) -> set:
    """
    Scan output directory for completed layout zip files.

    Args:
        output_dir: Directory containing _layoutN.zip files

    Returns:
        set: Layout indices that have been completed
    """
    completed = set()
    if not os.path.exists(output_dir):
        return completed

    for filename in os.listdir(output_dir):
        if filename.startswith("_layout") and filename.endswith(".zip"):
            try:
                # Extract layout number from filename like "_layout123.zip"
                layout_num = int(filename.replace("_layout", "").replace(".zip", ""))
                completed.add(layout_num)
            except ValueError:
                continue

    return completed


def load_layouts_and_inflows(data_dir: str):
    """
    Load layouts and inflows from .npz files.

    Args:
        data_dir: Directory containing layouts_metadata.npz and inflows_metadata.npz

    Returns:
        tuple: (layouts, layout_metadata, layout_inflows)
            layout_metadata includes split_indices dict if present in the file.
    """
    # Load layouts
    layouts_file = np.load(f"{data_dir}/layouts_metadata.npz", allow_pickle=True)
    n_layouts = int(layouts_file["n_layouts"])

    layouts = [layouts_file[f"layout_{i}"] for i in range(n_layouts)]
    layout_metadata = {
        "types": layouts_file["types"].tolist(),
        "spacings": layouts_file["spacings"].tolist(),
        "n_turbines": layouts_file["n_turbines"].tolist(),
    }

    # Load split indices if present (new format)
    if "train_indices" in layouts_file.files:
        layout_metadata["split_indices"] = {
            "shuffled_indices": layouts_file["shuffled_indices"],
            "train_indices": layouts_file["train_indices"],
            "val_indices": layouts_file["val_indices"],
            "test_indices": layouts_file["test_indices"],
        }
        # Also include split parameters if available
        if "split_seed" in layouts_file.files:
            layout_metadata["split_indices"]["split_seed"] = int(layouts_file["split_seed"])
        if "train_frac" in layouts_file.files:
            layout_metadata["split_indices"]["train_frac"] = float(layouts_file["train_frac"])
            layout_metadata["split_indices"]["val_frac"] = float(layouts_file["val_frac"])
            layout_metadata["split_indices"]["test_frac"] = float(layouts_file["test_frac"])

    # Load inflows
    inflows_file = np.load(f"{data_dir}/inflows_metadata.npz")
    n_inflow_layouts = int(inflows_file["n_layouts"])

    layout_inflows = [inflows_file[f"inflows_{i}"] for i in range(n_inflow_layouts)]

    return layouts, layout_metadata, layout_inflows
