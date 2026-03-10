"""
Dataset generation pipeline for GNO model training.

Generates wind farm layouts, simulates flow fields using PyWake, converts to graph format,
and preprocesses the data for training.
"""

import gc
import logging
import os
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from run_pywake import create_grid_for_layout, simulate_farm
from utils.graph_io import save_graphs_to_zip
from utils.inflow_generator import IEC_61400_1_2019_class_interpreter, InflowGenerator
from utils.layout_generator import PLayGen, layout_fits_in_bounds
from utils.preprocessing_utils import run_standard_preprocessing
from utils.pywake_utils import (
    DEFAULT_TO_GRAPH_KWS,
    create_layout_stats_dict,
    get_turbine_settings,
)
from utils.resume import get_completed_layouts, load_layouts_and_inflows

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    target_samples: int = 5000
    inflows_per_layout: int = 10
    n_turbines_range: tuple = (20, 100)
    spacing_range: tuple = (4, 10)
    layout_types: list = None
    layout_probs: list = None
    grid_density: int = 3
    num_cpu: int = None
    output_dir: str = "./data/large_graphs_nodes_2_v2"
    # Layout bounds (in rotor diameters D)
    # x_max: ±63.5D means total streamwise extent of 127D
    # y_max: ±127D means total crosswind extent of 254D
    layout_x_max: float = 63.5
    layout_y_max: float = 127.0

    def __post_init__(self):
        if self.layout_types is None:
            self.layout_types = [
                "cluster",
                "single string",
                "multiple string",
                "parallel string",
            ]
        if self.layout_probs is None:
            self.layout_probs = [0.4, 0.2, 0.2, 0.2]
        if self.num_cpu is None:
            self.num_cpu = int(os.cpu_count() / 2)


def generate_layouts(config: DatasetConfig, max_retries: int = 10):
    """
    Generate wind farm layouts with specified parameters.

    Args:
        config: Dataset configuration
        max_retries: Maximum attempts to generate a layout that fits within bounds

    Returns:
        tuple: (layouts, layout_metadata) where metadata contains types and spacings
    """
    logger.info(f"Generating {config.target_samples} wind farm layouts...")
    logger.info(
        f"Layout bounds: ±{config.layout_x_max}D streamwise, ±{config.layout_y_max}D crosswind"
    )

    extra_samples = int(config.target_samples * 0.1)
    total_samples = config.target_samples + extra_samples

    # Determine number of layouts per type, ensuring we generate exactly total_samples
    type_counts = (np.array(config.layout_probs) * total_samples).astype(int)
    # Add remaining samples to maintain exact count (distribute rounding errors)
    remaining = total_samples - type_counts.sum()
    if remaining > 0:
        # Add remaining samples to types with highest fractional parts
        fractional_parts = (np.array(config.layout_probs) * total_samples) - type_counts
        indices = np.argsort(fractional_parts)[-remaining:]
        type_counts[indices] += 1

    # Generate layouts
    layout_generator = PLayGen(D=1)
    layouts = []
    layout_types = []
    layout_spacings = []
    layout_n_turbines = []
    failed = 0
    out_of_bounds = 0

    for type_, count in zip(config.layout_types, type_counts):
        # Get turbine count range for this layout type
        if hasattr(config, "n_turbines_range_per_type"):
            n_min, n_max = config.n_turbines_range_per_type[type_]
        else:
            n_min, n_max = config.n_turbines_range

        # Get spacing range for this layout type
        if hasattr(config, "spacing_range_per_type"):
            s_min, s_max = config.spacing_range_per_type[type_]
        else:
            s_min, s_max = config.spacing_range

        for _ in range(count):
            if len(layouts) >= config.target_samples:
                break

            # Try to generate a layout that fits within bounds
            for attempt in range(max_retries):
                # Sample turbine count uniformly within range for this type
                n_wt = np.random.randint(n_min, n_max + 1)

                # Sample spacing
                spacing = np.random.uniform(s_min, s_max)

                layout_generator.set_N_turbs(n_wt)
                layout_generator.set_spacing(spacing)
                layout_generator.set_layout_style(type_)

                try:
                    layout = layout_generator()

                    # Check if layout fits within bounds
                    if layout_fits_in_bounds(
                        layout, x_max=config.layout_x_max, y_max=config.layout_y_max
                    ):
                        layouts.append(layout)
                        layout_types.append(type_)
                        layout_spacings.append(spacing)
                        layout_n_turbines.append(n_wt)
                        break  # Success, exit retry loop
                    else:
                        out_of_bounds += 1
                        if attempt == max_retries - 1:
                            logger.debug(
                                f"Layout exceeded bounds after {max_retries} attempts: "
                                f"type={type_}, n_wt={n_wt}, spacing={spacing:.1f}D"
                            )
                            failed += 1

                except Exception as e:
                    logger.debug(f"Failed to generate layout: {e}")
                    if attempt == max_retries - 1:
                        failed += 1

    logger.info(
        f"Generated {len(layouts)} layouts ({failed} failed, {out_of_bounds} out-of-bounds retries)"
    )

    metadata = {
        "types": layout_types,
        "spacings": layout_spacings,
        "n_turbines": layout_n_turbines,
    }

    return layouts, metadata


def generate_inflows(layouts, config: DatasetConfig):
    """
    Generate inflow conditions for all layouts.

    Args:
        layouts: List of layout arrays
        config: Dataset configuration

    Returns:
        list: Inflow arrays for each layout
    """
    logger.info(f"Generating {config.inflows_per_layout} inflow conditions per layout...")

    # Get turbine settings using shared utility
    turbine_settings, _, _ = get_turbine_settings()

    # Generate inflows
    inflow_settings = IEC_61400_1_2019_class_interpreter("I", "B")
    inflow_gen = InflowGenerator(
        inflow_settings=inflow_settings,
        turbine_settings=turbine_settings,
        ti_max=0.8,  # Maximum TI for capped method
    )

    n_samples = config.inflows_per_layout * len(layouts)
    inflows = inflow_gen.generate_inflows(
        n_samples,
        output_type="array",
        ti_method="Dimitrov_capped",  # Use capped method to avoid pileup at max
    )
    layout_inflows = np.split(inflows, len(layouts))

    logger.info(f"Generated {n_samples} total inflow conditions")

    return layout_inflows


def _prepare_simulation_data(
    layout_idx: int,
    layout: np.ndarray,
    inflows: np.ndarray,
    layout_metadata: dict,
    config: DatasetConfig,
    turbine_diameter: float,
    base_to_graph_kws: dict,
) -> dict:
    """Prepare simulation data for a single layout."""
    inflow_dict = {"u": inflows[:, 0], "ti": inflows[:, 1]}

    # Use shared utility for layout stats
    layout_stats_dict = create_layout_stats_dict(
        n_wt=layout.shape[0],
        layout_type=layout_metadata["types"][layout_idx],
        wt_spacing=layout_metadata["spacings"][layout_idx],
    )

    grid = create_grid_for_layout(
        layout,
        turbine_diameter=turbine_diameter,
        grid_density=config.grid_density,
        x_upstream=10.0,
        x_downstream=389.0,
        y_margin=5.0,
    )

    return {
        "inflow_dict": inflow_dict,
        "positions": layout * turbine_diameter,
        "grid": grid,
        "to_graph_kws": dict(base_to_graph_kws, **layout_stats_dict),
        "wake_config": getattr(config, "wake_config", None),
    }


def run_pywake_simulations(
    layouts,
    layout_metadata,
    layout_inflows,
    config: DatasetConfig,
    completed_layouts: set = None,
    sequential: bool = False,
):
    """
    Run PyWake simulations and convert to graphs.

    Args:
        layouts: List of layout arrays
        layout_metadata: Dictionary with layout types and spacings
        layout_inflows: Inflow conditions for each layout
        config: Dataset configuration
        completed_layouts: Set of layout indices to skip (for resume mode)
        sequential: If True, process one layout at a time (lower memory, slower)

    Returns:
        tuple: (original_trunk_shape, total_graphs_generated)
    """
    if completed_layouts is None:
        completed_layouts = set()

    n_to_skip = len(completed_layouts)
    if n_to_skip > 0:
        logger.info(f"Skipping {n_to_skip} already completed layouts")

    logger.info("Running PyWake simulations and converting to graphs...")
    if sequential:
        logger.info("Using sequential processing (memory-safe mode)")

    # Get turbine diameter using shared utility
    _, _, D = get_turbine_settings()

    # Use shared default to_graph kwargs
    base_to_graph_kws = DEFAULT_TO_GRAPH_KWS

    original_trunk_shape = None
    total_graphs = 0
    batch_data = []
    layout_indices = []

    for i, (layout, inflows) in enumerate(
        tqdm(
            zip(layouts, layout_inflows),
            total=len(layouts),
            desc="Processing layouts",
        )
    ):
        if i in completed_layouts:
            continue

        sim_data = _prepare_simulation_data(
            i, layout, inflows, layout_metadata, config, D, base_to_graph_kws
        )

        if sequential:
            # Process immediately
            graphs, trunk_shape = simulate_farm(
                sim_data["inflow_dict"],
                sim_data["positions"],
                sim_data["grid"],
                convert_to_graph=True,
                to_graph_kws=sim_data["to_graph_kws"],
                wake_config=sim_data["wake_config"],
            )

            if original_trunk_shape is None:
                original_trunk_shape = trunk_shape

            save_graphs_to_zip(graphs, i, config.output_dir)
            total_graphs += len(graphs)

            # Aggressive cleanup to prevent memory buildup
            del graphs, trunk_shape, sim_data
            gc.collect()

            # Force full garbage collection periodically
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1} layouts, forcing full GC")
                gc.collect(2)  # Full collection including oldest generation

        else:
            # Accumulate for batch processing
            batch_data.append(sim_data)
            layout_indices.append(i)

            # Process batch when full or at end
            if len(batch_data) >= config.num_cpu or i == len(layouts) - 1:
                if not batch_data:
                    continue

                # Use max_nbytes=1 to force fresh workers (prevents memory buildup)
                outputs = Parallel(n_jobs=config.num_cpu, max_nbytes=1)(
                    delayed(simulate_farm)(
                        data["inflow_dict"],
                        data["positions"],
                        data["grid"],
                        convert_to_graph=True,
                        to_graph_kws=data["to_graph_kws"],
                        wake_config=data["wake_config"],
                    )
                    for data in batch_data
                )

                for layout_num, (graphs, trunk_shape) in zip(layout_indices, outputs):
                    if original_trunk_shape is None:
                        original_trunk_shape = trunk_shape

                    save_graphs_to_zip(graphs, layout_num, config.output_dir)
                    total_graphs += len(graphs)
                    del graphs  # Free memory immediately after saving

                del batch_data, layout_indices, outputs
                batch_data = []
                layout_indices = []
                gc.collect()
                logger.debug(f"Garbage collection after batch ending at layout {i}")

    logger.info(f"Generated {total_graphs} total graphs")

    return original_trunk_shape, total_graphs


# Note: save_graphs_to_zip is imported from utils.graph_io
# Note: run_standard_preprocessing is imported from utils.preprocessing_utils


def save_layouts_and_inflows(
    layouts,
    layout_metadata,
    layout_inflows,
    output_dir: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    split_seed: int = 42,
):
    """
    Save layouts and inflows to .npz format for sharing and reuse.

    Includes pre-computed train/val/test split indices to ensure reproducible
    splits across different users and runs.

    Args:
        layouts: List of layout arrays
        layout_metadata: Dictionary with layout types, spacings, etc.
        layout_inflows: List of inflow arrays for each layout
        output_dir: Directory to save the files
        train_frac: Fraction of data for training (default: 0.6)
        val_frac: Fraction of data for validation (default: 0.2)
        test_frac: Fraction of data for testing (default: 0.2)
        split_seed: Random seed for reproducibility (default: 42)
    """
    from utils.preprocessing_utils import generate_split_indices

    logger.info("Saving layouts and inflows for sharing...")

    # Prepare layout data
    layout_data = {f"layout_{i}": layout for i, layout in enumerate(layouts)}
    layout_data["types"] = np.array(layout_metadata["types"], dtype=object)
    layout_data["spacings"] = np.array(layout_metadata["spacings"])
    layout_data["n_turbines"] = np.array(layout_metadata["n_turbines"])
    layout_data["n_layouts"] = len(layouts)

    # Generate and add split indices
    split_data = generate_split_indices(
        n_layouts=len(layouts),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=split_seed,
    )
    layout_data["shuffled_indices"] = split_data["shuffled_indices"]
    layout_data["train_indices"] = split_data["train_indices"]
    layout_data["val_indices"] = split_data["val_indices"]
    layout_data["test_indices"] = split_data["test_indices"]
    layout_data["split_seed"] = split_seed
    layout_data["train_frac"] = train_frac
    layout_data["val_frac"] = val_frac
    layout_data["test_frac"] = test_frac

    logger.info(
        f"Split indices saved: {len(split_data['train_indices'])} train, "
        f"{len(split_data['val_indices'])} val, {len(split_data['test_indices'])} test"
    )

    # Save layouts
    layouts_path = os.path.join(output_dir, "layouts_metadata.npz")
    np.savez(layouts_path, **layout_data)
    logger.info(f"Layouts saved to: {layouts_path}")

    # Prepare inflow data
    inflow_data = {f"inflows_{i}": inflow for i, inflow in enumerate(layout_inflows)}
    inflow_data["n_layouts"] = len(layout_inflows)
    inflow_data["inflows_per_layout"] = layout_inflows[0].shape[0] if layout_inflows else 0

    # Save inflows
    inflows_path = os.path.join(output_dir, "inflows_metadata.npz")
    np.savez(inflows_path, **inflow_data)
    logger.info(f"Inflows saved to: {inflows_path}")

    # Save human-readable split info JSON for easy sharing
    import json

    def build_layout_info(idx):
        """Build layout info dict for a given index."""
        return {
            "layout_idx": int(idx),
            "type": layout_metadata["types"][idx],
            "spacing": float(layout_metadata["spacings"][idx]),
            "n_turbines": int(layout_metadata["n_turbines"][idx]),
        }

    split_info = {
        "description": "Train/val/test split assignments for wind farm layouts",
        "split_parameters": {
            "seed": split_seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        },
        "summary": {
            "total_layouts": len(layouts),
            "n_train": len(split_data["train_indices"]),
            "n_val": len(split_data["val_indices"]),
            "n_test": len(split_data["test_indices"]),
        },
        "train": [build_layout_info(i) for i in split_data["train_indices"]],
        "val": [build_layout_info(i) for i in split_data["val_indices"]],
        "test": [build_layout_info(i) for i in split_data["test_indices"]],
    }

    split_info_path = os.path.join(output_dir, "split_info.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Split info saved to: {split_info_path}")

    logger.info("Layouts and inflows saved successfully")


def get_config(config_name: str) -> DatasetConfig:
    """
    Get dataset configuration by name.

    Args:
        config_name: Configuration name ('turbopark10_test' or 'turbopark250')

    Returns:
        DatasetConfig: Configuration object
    """
    if config_name == "turbopark10_test":
        config = DatasetConfig(
            target_samples=10,
            inflows_per_layout=4,
            n_turbines_range=(5, 300),
            spacing_range=(3, 10),
            grid_density=1,
            output_dir="./data/turbopark_10layouts_test",
            layout_types=["single string", "multiple string", "parallel string", "cluster"],
            layout_probs=[0.1, 0.2, 0.35, 0.35],
        )
        # TurbOPark wake configuration using official Nygaard_2022 implementation
        config.wake_config = {"use_nygaard_2022": True}
    elif config_name == "turbopark250":
        config = DatasetConfig(
            target_samples=250,
            inflows_per_layout=4,  # Changed from 20 to 4 inflows per graph
            n_turbines_range=(5, 300),
            spacing_range=(3, 10),
            grid_density=2,  # Reduced from 3 to limit memory usage
            num_cpu=2,  # Reduced from 4 - large layouts need ~1GB+ per worker
            output_dir="./data/turbopark_250layouts",
            layout_types=["single string", "multiple string", "parallel string", "cluster"],
            layout_probs=[0.1, 0.2, 0.35, 0.35],
        )
        # TurbOPark wake configuration using official Nygaard_2022 implementation
        config.wake_config = {"use_nygaard_2022": True}
    elif config_name == "turbopark2500":
        # Large-scale dataset for phase 1 pretraining
        # 2500 layouts × 4 inflows = 10,000 graphs
        config = DatasetConfig(
            target_samples=2500,
            inflows_per_layout=4,
            n_turbines_range=(5, 300),
            spacing_range=(3, 10),
            grid_density=2,  # Balance between resolution and memory
            num_cpu=2,  # Conservative for memory-intensive large layouts
            output_dir="./data/turbopark_2500layouts",
            layout_types=["single string", "multiple string", "parallel string", "cluster"],
            layout_probs=[0.1, 0.2, 0.35, 0.35],
        )
        # TurbOPark wake configuration using official Nygaard_2022 implementation
        config.wake_config = {"use_nygaard_2022": True}
    else:
        raise ValueError(
            f"Unknown config: {config_name}. "
            "Available: 'turbopark10_test', 'turbopark250', 'turbopark2500'"
        )

    # Turbine count ranges per layout type (same for both configs)
    config.n_turbines_range_per_type = {
        "single string": (5, 30),
        "multiple string": (5, 150),
        "parallel string": (5, 300),
        "cluster": (5, 300),
    }

    # Optional: Spacing ranges per layout type (in rotor diameters D)
    # Uncomment to use different spacing per type instead of uniform spacing_range
    # config.spacing_range_per_type = {
    #     "single string": (4, 8),
    #     "multiple string": (4, 6),
    #     "parallel string": (4, 6),
    #     "cluster": (4, 6),
    # }

    return config


def main():
    """Main dataset generation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate wind farm datasets with PyWake")
    parser.add_argument(
        "--config",
        type=str,
        default="turbopark10_test",
        choices=["turbopark10_test", "turbopark250", "turbopark2500"],
        help="Configuration to use (default: turbopark10_test)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation from saved layouts/inflows, skipping completed layouts",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process layouts sequentially (lower memory, slower)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing step (useful when resuming partial generation)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (overrides config default)",
    )
    args = parser.parse_args()

    logger.info("Starting dataset generation pipeline")
    logger.info(f"Configuration: {args.config}")
    if args.resume:
        logger.info("Resume mode: will load layouts/inflows from NPZ and skip completed")
    if args.sequential:
        logger.info("Sequential mode: processing one layout at a time (memory-safe)")

    # Get configuration
    config = get_config(args.config)

    # Override num_cpu if specified
    if args.num_workers is not None:
        config.num_cpu = args.num_workers
        logger.info(f"Using {config.num_cpu} parallel workers (override)")

    os.makedirs(config.output_dir, exist_ok=True)

    # Check for resume mode
    completed_layouts = set()
    if args.resume:
        # Load existing layouts and inflows from NPZ files
        layouts_path = os.path.join(config.output_dir, "layouts_metadata.npz")
        inflows_path = os.path.join(config.output_dir, "inflows_metadata.npz")

        if not os.path.exists(layouts_path) or not os.path.exists(inflows_path):
            logger.error(
                f"Cannot resume: layouts_metadata.npz or inflows_metadata.npz not found in {config.output_dir}"
            )
            logger.error("Run without --resume first to generate layouts and inflows")
            return

        logger.info(f"Loading layouts and inflows from {config.output_dir}")
        layouts, layout_metadata, layout_inflows = load_layouts_and_inflows(config.output_dir)
        logger.info(f"Loaded {len(layouts)} layouts with {layout_inflows[0].shape[0]} inflows each")

        # Find completed layouts
        completed_layouts = get_completed_layouts(config.output_dir)
        logger.info(
            f"Found {len(completed_layouts)} completed layouts: "
            f"{sorted(completed_layouts)[:10]}{'...' if len(completed_layouts) > 10 else ''}"
        )

        if len(completed_layouts) >= len(layouts):
            logger.info("All layouts already completed!")
            if not args.skip_preprocess:
                # Still run preprocessing in case it wasn't done
                run_standard_preprocessing(config.output_dir, None)
            return

    else:
        # Fresh generation: create new layouts and inflows
        # Step 1: Generate layouts
        layouts, layout_metadata = generate_layouts(config)

        # Step 2: Generate inflows
        layout_inflows = generate_inflows(layouts, config)

        # Step 2.5: Save layouts and inflows for sharing
        save_layouts_and_inflows(layouts, layout_metadata, layout_inflows, config.output_dir)

    # Step 3: Run PyWake simulations and create graphs
    original_trunk_shape, total_graphs = run_pywake_simulations(
        layouts,
        layout_metadata,
        layout_inflows,
        config,
        completed_layouts=completed_layouts,
        sequential=args.sequential,
    )

    # Step 4: Preprocess data
    if not args.skip_preprocess:
        run_standard_preprocessing(config.output_dir, original_trunk_shape)
    else:
        logger.info("Skipping preprocessing (--skip-preprocess flag set)")

    logger.info("=" * 80)
    logger.info("Dataset generation complete!")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Total layouts: {len(layouts)}")
    logger.info(f"Total graphs: {total_graphs}")
    if not args.resume:
        logger.info("Layouts/inflows saved for sharing and reuse")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
