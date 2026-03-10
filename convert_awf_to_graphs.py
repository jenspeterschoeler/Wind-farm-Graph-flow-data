"""
Convert AWF Database to Graph Format

This script converts the AWF (Aventa Wind Farm) database to a graph-based format
compatible with the GNO model, matching the format created by run_pywake.py.

The script performs both conversion and preprocessing in one step:
1. Converts AWF database to graph format
2. Computes statistics and applies min-max scaling
3. Splits into train/val/test sets

Usage:
    python convert_awf_to_graphs.py --database data/awf_database.nc --output data/awf_graphs --max-layouts 10
"""

import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from to_graph import to_graph
from utils.preprocessing_utils import (
    create_metadata_npz_from_split,
    export_split_to_json,
    generate_split_indices,
    load_split_from_json,
    run_standard_preprocessing,
)
from utils.pywake_utils import create_layout_stats_dict, get_turbine_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Get AWF turbine diameter using shared utility
_, _, AWF_TURBINE_DIAMETER = get_turbine_settings()  # meters


def _convert_single_flowcase(
    dataset: xr.Dataset,
    layout_idx: int,
    flowcase_idx: int,
    x_upstream_D: float = 10.0,
    y_margin_D: float = 5.0,
):
    """
    Convert one flowcase to graph format with grid aligned to run_pywake.py.

    Args:
        dataset: AWF xarray Dataset
        layout_idx: Layout index
        flowcase_idx: Flowcase index
        x_upstream_D: Upstream extent in diameters (default: 10)
        y_margin_D: Lateral margin in diameters (default: 5)

    Note:
        AWF database coordinates (x, y, wt_x, wt_y) are in DIAMETERS, not meters.
        This function clips in diameter units, then converts to meters for output.
        Probe subsampling is not applied here - the pipeline handles variable probe
        counts through dynamic batching.
    """
    fc = dataset.isel(layout=layout_idx, flowcase=flowcase_idx)

    nwt = int(fc.Nwt.values)
    wt_x_D = fc.wt_x.values[:nwt]  # In diameters
    wt_y_D = fc.wt_y.values[:nwt]  # In diameters

    # Extract effective wind speed at each turbine (wseff)
    wseff = dataset.lut_wseff.isel(layout=layout_idx, flowcase=flowcase_idx).values[:nwt]

    # Get AWF grid (in diameters)
    x_grid_full_D = fc.x.values
    y_grid_full_D = fc.y.values

    # Calculate wind farm extent (in diameters)
    x_min_wf_D = wt_x_D.min()
    y_min_wf_D = wt_y_D.min()
    y_max_wf_D = wt_y_D.max()

    # Define grid bounds centered on wind farm (in diameters, like run_pywake.py)
    x_min_desired_D = x_min_wf_D - x_upstream_D
    y_min_desired_D = y_min_wf_D - y_margin_D
    y_max_desired_D = y_max_wf_D + y_margin_D

    # Filter grid points within desired bounds (no downstream clipping)
    x_mask = x_grid_full_D >= x_min_desired_D
    y_mask = (y_grid_full_D >= y_min_desired_D) & (y_grid_full_D <= y_max_desired_D)

    x_grid_clipped_D = x_grid_full_D[x_mask]
    y_grid_clipped_D = y_grid_full_D[y_mask]

    # Create meshgrid and flatten (still in diameters)
    x_mesh_D, y_mesh_D = np.meshgrid(x_grid_clipped_D, y_grid_clipped_D, indexing="ij")
    probe_x_D = x_mesh_D.flatten()
    probe_y_D = y_mesh_D.flatten()

    # Extract velocity field for clipped grid
    U_full = fc.U.values  # shape: (len(x), len(y))
    U_clipped = U_full[x_mask, :][:, y_mask]
    U_flat = U_clipped.flatten()

    # AWF stores velocities as U/U_inf ratios - denormalize to absolute m/s
    ws_inf = float(fc.ws_inf.values)
    U_flat = U_flat * ws_inf

    # Convert from diameters to meters for GNO model
    wt_x_m = wt_x_D * AWF_TURBINE_DIAMETER
    wt_y_m = wt_y_D * AWF_TURBINE_DIAMETER
    probe_x_m = probe_x_D * AWF_TURBINE_DIAMETER
    probe_y_m = probe_y_D * AWF_TURBINE_DIAMETER

    # Positions should only be turbine positions (probes go in trunk_inputs)
    positions = np.c_[wt_x_m, wt_y_m]

    # Node features should only be wind turbine wind speeds (wseff)
    node_features = wseff.reshape(-1, 1)

    # Use shared utility for layout stats
    layout_stats = create_layout_stats_dict(n_wt=nwt)

    graph = to_graph(
        points=positions,
        connectivity="delaunay",
        add_edge="cartesian",
        node_features=node_features,
        global_features=np.array([ws_inf, float(fc.ti_inf.values)]),
        trunk_inputs=np.c_[probe_x_m, probe_y_m],
        output_features=U_flat.reshape(-1, 1),  # Only wind speed, matching PyWake format
        rel_wd=None,
        **layout_stats,
    )

    return graph


def _save_layout_graphs_to_zip(graphs: list, zip_path: str):
    """Save graphs for a single layout to a zip file."""
    import io

    with ZipFile(zip_path, "w") as zf:
        for i, graph in enumerate(graphs):
            buffer = io.BytesIO()
            torch.save(graph, buffer)
            buffer.seek(0)
            zf.writestr(f"flowcase_{i:04d}.pt", buffer.read())


def convert_awf_to_graphs(
    database_path: str,
    output_dir: str,
    max_layouts: int | None = None,
    x_upstream_D: float = 10.0,
    y_margin_D: float = 5.0,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    skip_preprocessing: bool = False,
    split_file: str | None = None,
):
    """
    Convert AWF database to graph format and preprocess for training.

    This function performs the complete pipeline:
    1. Converts AWF database to graph format matching run_pywake.py output
    2. Computes dataset statistics
    3. Applies min-max scaling (run4 method)
    4. Splits into train/val/test sets

    Args:
        database_path: Path to AWF NetCDF file
        output_dir: Output directory for graph datasets
        max_layouts: Optional limit on number of layouts
        x_upstream_D: Upstream extent in diameters (default: 10)
        y_margin_D: Lateral margin in diameters (default: 5)
        train_size: Fraction of data for training (default: 0.6)
        val_size: Fraction of data for validation (default: 0.2)
        test_size: Fraction of data for testing (default: 0.2)
        skip_preprocessing: If True, only convert without preprocessing
        split_file: Path to a JSON file with pre-defined train/val/test
            split indices. If provided, the split is loaded and used instead
            of generating a new one. In either case, the split is saved to
            ``layouts_metadata.npz`` (for preprocessing) and
            ``split_info.json`` (for portability).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading AWF database from {database_path}")
    dataset = xr.open_dataset(database_path)

    n_layouts = len(dataset.layout)
    n_flowcases = len(dataset.flowcase)

    if max_layouts:
        n_layouts = min(n_layouts, max_layouts)

    logger.info(f"Processing {n_layouts} layouts with {n_flowcases} flowcases each")
    logger.info(
        f"Grid bounds: {x_upstream_D}D upstream, full downstream extent, "
        f"{y_margin_D}D lateral margin"
    )

    trunk_shape = None  # Will be determined from first graph
    total_graphs = 0

    for layout_idx in tqdm(range(n_layouts), desc="Converting layouts"):
        layout_graphs = []
        for flowcase_idx in range(n_flowcases):
            graph = _convert_single_flowcase(
                dataset,
                layout_idx,
                flowcase_idx,
                x_upstream_D,
                y_margin_D,
            )
            layout_graphs.append(graph)

            # Capture trunk shape from first graph for preprocessing
            if trunk_shape is None:
                trunk_shape = tuple(graph.trunk_inputs.shape)
                logger.info(f"Detected trunk shape: {trunk_shape}")

        # Save this layout's graphs to a separate zip file (matching pre_process.py format)
        zip_path = output_path / f"_layout{layout_idx}.zip"
        _save_layout_graphs_to_zip(layout_graphs, str(zip_path))
        total_graphs += len(layout_graphs)

    dataset.close()

    logger.info(f"Conversion complete: {total_graphs} graphs saved across {n_layouts} layout files")
    logger.info(f"Output: {output_path}")

    # --- Persist split indices (always, regardless of skip_preprocessing) ---
    if split_file is not None:
        logger.info(f"Loading pre-defined split from {split_file}")
        split = load_split_from_json(split_file)
        # Override train/val/test fractions from actual split counts
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        train_size = len(split["train"]) / total
        val_size = len(split["val"]) / total
        test_size = len(split["test"]) / total
    else:
        logger.info("Generating new train/val/test split")
        raw = generate_split_indices(
            n_layouts=n_layouts,
            train_frac=train_size,
            val_frac=val_size,
            test_frac=test_size,
        )
        split = {
            "train": raw["train_indices"].tolist(),
            "val": raw["val_indices"].tolist(),
            "test": raw["test_indices"].tolist(),
        }

    # Write layouts_metadata.npz (consumed by pre_process.py)
    create_metadata_npz_from_split(split, str(output_path))

    # Write portable JSON (for sharing the split with collaborators)
    json_out = output_path / "split_info.json"
    export_split_to_json(
        {
            "train_indices": split["train"],
            "val_indices": split["val"],
            "test_indices": split["test"],
        },
        str(json_out),
    )

    # Run preprocessing if not skipped
    if not skip_preprocessing:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Starting Preprocessing")
        logger.info("=" * 80)
        logger.info(f"Train/Val/Test split: {train_size:.0%} / {val_size:.0%} / {test_size:.0%}")
        logger.info(f"Trunk shape (auto-detected): {trunk_shape}")

        try:
            # Use shared preprocessing utility
            run_standard_preprocessing(
                output_dir=str(output_path),
                original_trunk_shape=trunk_shape,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
            )

            logger.info("")
            logger.info("=" * 80)
            logger.info("Preprocessing Complete!")
            logger.info("=" * 80)
            logger.info(f"Processed dataset: {output_path}")
            logger.info("  - train_pre_processed/")
            logger.info("  - val_pre_processed/")
            logger.info("  - test_pre_processed/")
            logger.info("  - stats.json")
            logger.info("  - scale_stats.json")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert AWF database to graph dataset and preprocess for GNO training"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/awf_database.nc",
        help="Path to AWF NetCDF file",
    )
    parser.add_argument("--output", type=str, default="data/awf_graphs", help="Output directory")
    parser.add_argument("--max-layouts", type=int, default=None, help="Limit layouts (optional)")
    parser.add_argument(
        "--x-upstream-D",
        type=float,
        default=10.0,
        help="Upstream extent in diameters (default: 10)",
    )
    parser.add_argument(
        "--y-margin-D",
        type=float,
        default=5.0,
        help="Lateral margin in diameters (default: 5)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Fraction for training set (default: 0.6)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction for validation set (default: 0.2)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step (only convert)",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to a JSON file with pre-defined train/val/test split indices",
    )

    args = parser.parse_args()

    success = convert_awf_to_graphs(
        database_path=args.database,
        output_dir=args.output,
        max_layouts=args.max_layouts,
        x_upstream_D=args.x_upstream_D,
        y_margin_D=args.y_margin_D,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        skip_preprocessing=args.skip_preprocessing,
        split_file=args.split_file,
    )

    if success:
        logger.info("")
        logger.info("All done! Dataset is ready for training.")
    else:
        logger.error("Conversion or preprocessing failed.")
