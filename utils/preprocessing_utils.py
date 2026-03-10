"""
Preprocessing utility functions for data generation.

This module provides wrapper functions around the main pre_process module
with standard default parameters, plus utilities for exporting/importing
train/val/test split indices as JSON.
"""

import json
import logging
from pathlib import Path

import numpy as np

from pre_process import pre_process

logger = logging.getLogger(__name__)

# Standard preprocessing parameters
DEFAULT_TRAIN_SIZE = 0.6
DEFAULT_VAL_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SCALING_METHOD = "run4"
DEFAULT_SPLIT_SEED = 42


def generate_split_indices(
    n_layouts: int,
    train_frac: float = DEFAULT_TRAIN_SIZE,
    val_frac: float = DEFAULT_VAL_SIZE,
    test_frac: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    """
    Generate reproducible train/val/test split indices.

    Uses a fixed seed (default 42) for reproducibility across different
    runs and users.

    Args:
        n_layouts: Total number of layouts to split
        train_frac: Fraction of data for training (default: 0.6)
        val_frac: Fraction of data for validation (default: 0.2)
        test_frac: Fraction of data for testing (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dict: Dictionary containing:
            - shuffled_indices: Full shuffle order of all indices
            - train_indices: Indices for training set
            - val_indices: Indices for validation set
            - test_indices: Indices for test set
    """
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(n_layouts)
    rng.shuffle(indices)

    n_train = int(np.floor(n_layouts * train_frac))
    n_val = int(np.floor(n_layouts * val_frac))

    return {
        "shuffled_indices": indices,
        "train_indices": indices[:n_train],
        "val_indices": indices[n_train : n_train + n_val],
        "test_indices": indices[n_train + n_val :],
    }


def load_split_from_json(json_path: str) -> dict:
    """
    Load a pre-defined train/val/test split from a JSON file.

    Expected JSON format::

        {
            "train": [3, 5, 7, ...],
            "val": [18, 20, 24, ...],
            "test": [0, 11, 12, ...]
        }

    Args:
        json_path: Path to the split JSON file.

    Returns:
        dict with keys "train", "val", "test", each a sorted list of ints.
    """
    with open(json_path) as f:
        data = json.load(f)

    return {
        "train": sorted(int(i) for i in data["train"]),
        "val": sorted(int(i) for i in data["val"]),
        "test": sorted(int(i) for i in data["test"]),
    }


def export_split_to_json(split: dict, output_path: str) -> None:
    """
    Export a train/val/test split to a JSON file.

    Args:
        split: dict with keys "train_indices", "val_indices", "test_indices"
            (numpy arrays or lists of ints).
        output_path: Destination JSON path.
    """

    def _to_list(arr):
        return sorted(int(i) for i in arr)

    payload = {
        "description": "Train/val/test split indices (layout numbers)",
        "summary": {
            "n_train": len(split["train_indices"]),
            "n_val": len(split["val_indices"]),
            "n_test": len(split["test_indices"]),
            "total_layouts": (
                len(split["train_indices"]) + len(split["val_indices"]) + len(split["test_indices"])
            ),
        },
        "train": _to_list(split["train_indices"]),
        "val": _to_list(split["val_indices"]),
        "test": _to_list(split["test_indices"]),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Exported split to {output_path}")


def create_metadata_npz_from_split(split: dict, output_dir: str) -> None:
    """
    Create a ``layouts_metadata.npz`` that ``pre_process.py`` will pick up.

    Args:
        split: dict with keys "train", "val", "test" (lists of ints).
        output_dir: Directory where the npz will be written.
    """
    train = np.array(sorted(split["train"]), dtype=np.int64)
    val = np.array(sorted(split["val"]), dtype=np.int64)
    test = np.array(sorted(split["test"]), dtype=np.int64)
    shuffled = np.concatenate([train, val, test])
    n_layouts = len(shuffled)

    out_path = Path(output_dir) / "layouts_metadata.npz"
    np.savez(
        out_path,
        shuffled_indices=shuffled,
        train_indices=train,
        val_indices=val,
        test_indices=test,
        n_layouts=n_layouts,
    )
    logger.info(f"Created {out_path}  (train={len(train)}, val={len(val)}, test={len(test)})")


def run_standard_preprocessing(
    output_dir: str,
    original_trunk_shape=None,
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    scaling_method: str = DEFAULT_SCALING_METHOD,
):
    """
    Run preprocessing with standard default parameters.

    This is a convenience wrapper around pre_process() that uses the standard
    train/val/test split (60/20/20) and run4 scaling method.

    Args:
        output_dir: Directory containing the generated graphs
        original_trunk_shape: Shape of the trunk output (for scaling)
        train_size: Fraction of data for training (default: 0.6)
        val_size: Fraction of data for validation (default: 0.2)
        test_size: Fraction of data for testing (default: 0.2)
        scaling_method: Scaling method to use (default: "run4")
    """
    logger.info("Running preprocessing: scaling and train/val/test split...")

    pre_process(
        output_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        scaling_method=scaling_method,
        original_trunk_shape=original_trunk_shape,
    )

    logger.info("Preprocessing complete")
