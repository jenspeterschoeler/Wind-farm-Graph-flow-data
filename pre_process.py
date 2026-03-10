"""Pre-processing pipeline for PyWake simulation data."""

import datetime
import io
import json
import logging
import os
import tempfile
from zipfile import ZipFile

import numba as nb
import numpy as np
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from to_graph import PyGTupleData

NestedListStr = str | list["NestedListStr"]
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")


def load_split_indices_from_metadata(data_path: str) -> dict | None:
    """
    Load pre-computed split indices from layouts_metadata.npz if available.

    This enables reproducible train/val/test splits that are consistent
    across different users and preprocessing runs.

    Args:
        data_path: Path to the dataset directory containing layouts_metadata.npz

    Returns:
        dict | None: Dictionary with train_indices, val_indices, test_indices arrays
            if split indices are present in the metadata file, None otherwise
            (for backward compatibility with old metadata files).
    """
    metadata_path = os.path.join(data_path, "layouts_metadata.npz")

    if not os.path.exists(metadata_path):
        return None

    metadata = np.load(metadata_path, allow_pickle=True)

    # Check if split indices are present (new format)
    if "train_indices" not in metadata.files:
        logger.info("No pre-computed split indices found in metadata (old format)")
        return None

    split_data = {
        "shuffled_indices": metadata["shuffled_indices"],
        "train_indices": metadata["train_indices"],
        "val_indices": metadata["val_indices"],
        "test_indices": metadata["test_indices"],
    }

    logger.info(
        f"Loaded pre-computed split indices: {len(split_data['train_indices'])} train, "
        f"{len(split_data['val_indices'])} val, {len(split_data['test_indices'])} test"
    )

    return split_data


class Torch_Geomtric_Dataset(Dataset):
    """Dataset class for the GraphFarmsOperator dataset.
    Inteded to load the data from the disc into a format that can be used with the Graph Operator DataLoader.
    """

    # TODO Consider wheter this should live in spo-operator-test as there now is 2 versions of the same class
    def __init__(self, root_path: str, in_mem: bool = True) -> None:
        super().__init__()
        self._root_path = root_path
        self.in_mem = in_mem
        # get the list of all the zip files and their contents

        # create a list of tuples with the zip file path and the contents

        # store the zip file and its contents in a single matrix
        self.zip_matrix = self._create_zip_matrix()
        all_zip_paths_repeated, all_zip_item_names = self._create_indexes()
        self.zip_paths_repeated = all_zip_paths_repeated
        self.zip_contents = all_zip_item_names

        # If data should be kept in memory, load it once
        if self.in_mem:
            self._cache = [
                self._open_single_content_in_zip((zip_path, zip_item))
                for zip_path, zip_item in zip(self.zip_paths_repeated, self.zip_contents)
            ]

    def _open_zip(self, zip_construct: list[str]):
        zip_path = zip_construct[0]
        zip_items = zip_construct[1:]
        content = ()
        with ZipFile(zip_path) as zf:
            for item in zip_items:
                with zf.open(item) as f:
                    stream = io.BytesIO(f.read())
                    data = torch.load(stream, weights_only=False)
                    content += (data,)
        return content

    def _open_single_content_in_zip(self, zip_construct: list[str]):
        zip_path = zip_construct[0]
        zip_item = zip_construct[1]
        with ZipFile(zip_path) as zf, zf.open(zip_item) as f:
            stream = io.BytesIO(f.read())
            data = torch.load(stream, weights_only=False)
        return data

    def _create_zip_matrix(self) -> NestedListStr:
        zip_list = [
            os.path.join(path, name)
            for path, subdirs, files in os.walk(self._root_path)
            for name in files
        ]
        zip_list = [zip_file for zip_file in zip_list if zip_file.endswith(".zip")]
        self.zip_list = zip_list

        zip_matrix = []
        # Extract file paths from each zip file and ensure each zip has exactly two components
        for zip_path in zip_list:
            if ".zip" in zip_path:
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_items = zip_ref.namelist()

                zip_matrix.append((zip_path, *zip_items))
        return zip_matrix

    def _create_indexes(self) -> tuple[np.ndarray, np.ndarray]:
        all_zip_paths_repeated = []
        all_zip_item_names = []
        for zip_construct in self.zip_matrix:
            zip_path, *zip_items = zip_construct
            repeated_zip_path = [zip_path] * len(zip_items)
            all_zip_paths_repeated += repeated_zip_path
            all_zip_item_names += zip_items
        return all_zip_paths_repeated, all_zip_item_names

    def __len__(self) -> int:
        return len(self.zip_paths_repeated)

    def __getitem__(self, idx: int) -> PyGTupleData:
        (self.zip_paths_repeated[idx], self.zip_contents[idx])

        if self.in_mem:
            # Return the preloaded item from cache.
            return self._cache[idx]
        else:
            # Read from disk on demand.
            zip_path = self.zip_paths_repeated[idx]
            zip_item = self.zip_contents[idx]
            return self._open_single_content_in_zip((zip_path, zip_item))


class online_stats_alg:
    """Online algorithm to compute mean, variance/std, min and max
    For the calculation of mean and variance, the Welford's online algorithm is used [1].
        [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, target_shape: int, vals_2d=False) -> None:
        # assert len(target_shape) == 1
        self.count = 0
        self.mean = np.zeros(target_shape)
        self.M2 = np.zeros(target_shape)
        self.min = np.ones((1, target_shape)) * 1e9
        self.max = np.ones((1, target_shape)) * -1e9
        self.vals_2d = vals_2d
        if self.vals_2d:
            self.update = self.update2d

            @nb.njit()  # the cummulatives are superslow therefore we use numba, also it cannot be rewritten as a list comprehension because of += operator
            def update_values(count, mean, M2, value):
                count += 1
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                M2 += delta * delta2
                return count, mean, M2

            self.update_2d_values = update_values

        else:
            self.update = self.update1d

    # For a new value new_value, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update1d(self, new_value: np.ndarray) -> None:
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

        self.min = np.min(
            np.concatenate([self.min, np.atleast_2d(new_value)], axis=0),
            axis=0,
            keepdims=True,
        )
        self.max = np.max(
            np.concatenate([self.max, np.atleast_2d(new_value)], axis=0),
            axis=0,
            keepdims=True,
        )

    def update2d(self, new_values: np.ndarray) -> None:
        for value in new_values:
            self.count, self.mean, self.M2 = self.update_2d_values(
                self.count, self.mean, self.M2, value
            )

        self.min = np.min(
            np.concatenate([self.min, np.atleast_2d(new_values)], axis=0),
            axis=0,
            keepdims=True,
        )
        self.max = np.max(
            np.concatenate([self.max, np.atleast_2d(new_values)], axis=0),
            axis=0,
            keepdims=True,
        )

    # Retrieve the mean, variance and sample variance from an aggregate
    def finalize(self) -> dict:
        if self.count < 2:
            return float("nan")
        else:
            mean = self.mean
            variance = self.M2 / self.count
            std = np.sqrt(variance)
            stats_dict = {
                "mean": mean,
                "variance": variance,
                "std": std,
                "min": np.squeeze(self.min),
                "max": np.squeeze(self.max),
            }
            return stats_dict


def compute_dataset_stats(
    dataset: Torch_Geomtric_Dataset, max_indexes: int, original_trunk_shape: tuple
) -> dict:
    # compute the stats

    random_graph = dataset[0]  # Assumes features are the same for all graphs
    random_trunk = random_graph.trunk_inputs
    random_output = random_graph.output_features

    stats = {
        "date_ISO8601": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "graph_size": {
            "max_n_nodes": 0,
            "max_n_edges": 0,
            "max_n_globals": 0,
        },
        "trunk_shape": original_trunk_shape,
    }

    osa_node = online_stats_alg(random_graph.node_features.shape[-1])
    osa_edge = online_stats_alg(random_graph.edge_attr.shape[-1])
    osa_global = online_stats_alg(random_graph.global_features.shape[-1])
    osa_trunk = online_stats_alg(random_trunk.shape[-1], vals_2d=True)
    osa_output = online_stats_alg(random_output.shape[-1], vals_2d=True)

    for i, graph in tqdm(enumerate(dataset)):
        trunk = graph.trunk_inputs
        output = graph.output_features
        # Loop to get the max values of the graph
        graph_shapes = {}
        for key, value in graph:
            if key in [
                "layout_type",
                "wt_spacing",
                "n_wt",
                "wake_model",
                "n_probes",
                "layout_idx",
                "flowcase_idx",
            ]:
                continue
            # Skip non-tensor attributes
            if not hasattr(value, "shape"):
                continue
            graph_shapes[key] = value.shape

        for key, stats_key in zip(
            ["node_features", "edge_attr", "global_features"],
            ["max_n_nodes", "max_n_edges", "max_n_globals"],
        ):
            shape = graph_shapes[key]
            if shape[0] > stats["graph_size"][stats_key]:
                stats["graph_size"][stats_key] = shape[0]

        for node in graph.node_features:
            node = node.numpy()
            osa_node.update(node)
        for edge in graph.edge_attr:
            edge = edge.numpy()
            osa_edge.update(edge)

        global_ = (
            graph.global_features.numpy().copy()
        )  # if not inside a loop it will be a reference, therefore .copy()
        osa_global.update(global_)

        osa_trunk.update(trunk.numpy())

        osa_output.update(output.numpy())

        if i == max_indexes:
            break

    stats["node_features"] = osa_node.finalize()
    stats["edge_features"] = osa_edge.finalize()
    stats["global_features"] = osa_global.finalize()
    stats["trunk"] = osa_trunk.finalize()
    stats["output"] = osa_output.finalize()
    stats["n_graphs"] = i + 1
    stats["max_indexes"] = max_indexes

    # convert to arrays to list for json serialization
    for key in stats:
        if isinstance(stats[key], dict):
            for sub_key in stats[key]:
                if isinstance(stats[key][sub_key], np.ndarray):
                    stats[key][sub_key] = stats[key][sub_key].tolist()
        else:
            if isinstance(stats[key], np.ndarray):
                stats[key] = stats[key].tolist()
    return stats


def retrieve_dataset_stats(
    dataset: Torch_Geomtric_Dataset, max_indexes: int, original_trunk_shape: tuple
) -> dict:
    path = dataset._root_path  # Access the private attribute of the dataset
    stats_path = os.path.join(path, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info(
            f"Dataset stats loaded from: {stats_path}, \nComputed on: {stats['date_ISO8601']}"
        )
    else:
        logger.info(f"No dataset stats found at: {stats_path}, computing stats...")
        stats = compute_dataset_stats(dataset, max_indexes, original_trunk_shape)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Dataset stats computed and saved to: {stats_path}")
    return stats


def append_globals_to_nodes(data):
    """
    Append global features to each node in the graph
    """
    assert data.n_node.shape[0] == 1
    n_nodes = int(data.n_node[0])
    # n_globals = data.n_edge[0]
    data.node_features = torch.cat(
        [data.node_features, data.global_features.repeat(n_nodes, 1)], dim=1
    )
    return data


# DEPRECATED: standard_scale (run1) is no longer supported
# Only run4 (min-max scaling with shared stats) is supported


def obtain_min_max_values(stats):
    """
    Obtain min-max values for run4 scaling method (min-max with shared stats and zero minimums).

    This is the only supported scaling method. Uses shared stats for distances and velocities
    to maintain physical relationships, with zero minimums for compatibility with residual
    connections and on-the-fly graph construction.
    """
    distance_min = torch.Tensor(
        stats["trunk"]["min"]
    ).min()  # Shared min and max for all distances maintains aspect ratio
    distance_max = torch.Tensor(stats["trunk"]["max"]).max()
    distance_range = distance_max - distance_min

    u_min = torch.Tensor([stats["global_features"]["min"][0]])
    u_max = torch.Tensor([stats["global_features"]["max"][0]])
    deficit_min = torch.Tensor([stats["output"]["min"]])
    deficit_max = torch.Tensor([stats["output"]["max"]])

    velocity_min = torch.min(u_min, deficit_min)
    velocity_max = torch.max(u_max, deficit_max)
    velocity_range = velocity_max - velocity_min

    ti_min = torch.Tensor([stats["global_features"]["min"][1]])
    ti_max = torch.Tensor([stats["global_features"]["max"][1]])
    ti_range = torch.Tensor([ti_max - ti_min])

    # Handle both 1D (wind turbine wind speed) and 3D (ws_eff, ti_eff, CT) node features
    node_min = stats["node_features"]["min"]
    node_max = stats["node_features"]["max"]
    if isinstance(node_min, int | float):
        node_min = [node_min]
        node_max = [node_max]

    if len(node_min) == 3:
        # Node features contain ws_eff, ti_eff, CT (legacy format)
        logger.info(
            "Node features contain 3 dimensions (ws_eff, ti_eff, CT). Positions are cartesian."
        )
        ws_eff_min = torch.Tensor([node_min[0]])
        ws_eff_max = torch.Tensor([node_max[0]])
        ti_eff_min = torch.Tensor([node_min[1]])
        ti_eff_max = torch.Tensor([node_max[1]])
        ct_min = torch.Tensor([node_min[2]])
        ct_range = torch.Tensor([node_max[2]]) - ct_min

        velocity_min = torch.min(velocity_min, ws_eff_min)
        velocity_max = torch.max(velocity_max, ws_eff_max)
        ti_min = torch.min(ti_min, ti_eff_min)
        ti_max = torch.max(ti_max, ti_eff_max)

        velocity_range = velocity_max - velocity_min
        ti_range = ti_max - ti_min
    elif len(node_min) == 1:
        # Node features contain only wind turbine wind speed (current format)
        logger.info(
            "Node features contain 1 dimension (turbine wind speed). Positions are cartesian."
        )
        wt_ws_min = torch.Tensor([node_min[0]])
        wt_ws_max = torch.Tensor([node_max[0]])
        wt_ws_max - wt_ws_min

        # Update velocity min/max to include node features
        velocity_min = torch.min(velocity_min, wt_ws_min)
        velocity_max = torch.max(velocity_max, wt_ws_max)
        velocity_range = velocity_max - velocity_min

        # For backward compatibility, set ct to None (not used in current format)
        ct_min = None
        ct_range = None
    else:
        raise ValueError(f"Unexpected node feature dimension: {len(node_min)}")

    scale_stats = {
        "distance": {"min": distance_min, "range": distance_range},
        "velocity": {"min": velocity_min, "range": velocity_range},
        "ti": {"min": ti_min, "range": ti_range},
        "ct": {"min": ct_min, "range": ct_range},
    }
    return scale_stats


def min_max_scale(data, scale_stats):
    """
    Min-max scale the dataset using run4 method (zero minimums).

    This sets all minimums to zero for compatibility with residual connections
    and on-the-fly graph construction (e.g., for probe setup).

    Assumes:
    - Global features: [wind_speed, turbulence_intensity]
    - Node features: Either [turbine_wind_speed] (1D, current) or [ws_eff, ti_eff, CT] (3D, legacy)
    - Positions are Cartesian coordinates
    """
    velocity_min = torch.tensor(scale_stats["velocity"]["min"])
    velocity_range = torch.tensor(scale_stats["velocity"]["range"])
    distance_min = torch.tensor(scale_stats["distance"]["min"])
    distance_range = torch.tensor(scale_stats["distance"]["range"])
    ti_min = torch.tensor(scale_stats["ti"]["min"])
    ti_range = torch.tensor(scale_stats["ti"]["range"])

    # CT stats only exist for legacy 3D node features format
    ct_min = scale_stats["ct"]["min"]
    ct_range = scale_stats["ct"]["range"]
    if ct_min is not None:
        ct_min = torch.tensor(ct_min)
        ct_range = torch.tensor(ct_range)

    # run4: Set minimums to zero for res-net compatibility and on-the-fly construction
    distance_min = torch.tensor([0.0])
    # velocity_min needs to match velocity_range shape (might be multi-channel)
    if velocity_range.dim() > 1 and velocity_range.shape[1] > 1:
        # Multi-channel output (e.g., [1, 2] for velocity and TKE)
        velocity_min = torch.zeros_like(velocity_range)
    else:
        velocity_min = torch.tensor([0.0])
    ti_min = torch.tensor([0.0])
    if ct_min is not None:
        ct_min = torch.tensor([0.0])

    # Build node_min and node_range based on actual node features
    # For 1 feature (turbine wind speed), use velocity stats
    # For 3 features (ws_eff, ti_eff, CT), use all three (legacy format)
    if data.node_features.shape[1] == 1:
        # Only turbine wind speed in node features (current format)
        # Use velocity stats since node feature is a wind speed
        if velocity_min.dim() > 1:
            node_min = (
                velocity_min[:, 0] if velocity_min.shape[1] > 0 else velocity_min.flatten()[0:1]
            )
            node_range = (
                velocity_range[:, 0]
                if velocity_range.shape[1] > 0
                else velocity_range.flatten()[0:1]
            )
        else:
            node_min = velocity_min
            node_range = velocity_range
    elif data.node_features.shape[1] == 3:
        # ws_eff, ti_eff, CT in node features
        # velocity_min/range might be 2D for output features, take first channel for ws
        if velocity_min.dim() > 1:
            velocity_min_scalar = (
                velocity_min[:, 0] if velocity_min.shape[1] > 0 else velocity_min.flatten()[0:1]
            )
            velocity_range_scalar = (
                velocity_range[:, 0]
                if velocity_range.shape[1] > 0
                else velocity_range.flatten()[0:1]
            )
        else:
            velocity_min_scalar = velocity_min
            velocity_range_scalar = velocity_range
        node_min = torch.cat([velocity_min_scalar, ti_min, ct_min], dim=0)
        node_range = torch.cat([velocity_range_scalar, ti_range, ct_range], dim=0)
    else:
        raise ValueError(f"Unexpected node feature dimension: {data.node_features.shape[1]}")

    # Global features are always [velocity, ti]
    # Handle velocity_min which might be 2D (for multiple output channels)
    if velocity_min.dim() > 1 and velocity_min.shape[1] > 1:
        # If velocity has multiple channels (e.g., [1, 2] for U and tke),
        # take first channel for global feature (U only)
        vel_min_for_global = velocity_min[:, 0:1].flatten()  # Shape [1]
        vel_range_for_global = velocity_range[:, 0:1].flatten()  # Shape [1]
    else:
        # Single channel or already 1D
        vel_min_for_global = velocity_min.flatten()
        vel_range_for_global = velocity_range.flatten()

    # Ensure ti_min and ti_range are also 1D
    ti_min_flat = ti_min.flatten()
    ti_range_flat = ti_range.flatten()

    global_min = torch.cat([vel_min_for_global, ti_min_flat], dim=0)
    global_range = torch.cat([vel_range_for_global, ti_range_flat], dim=0)

    data.output_features = (data.output_features - velocity_min) / velocity_range
    data.node_features = (
        data.node_features - node_min
    ) / node_range  #! NOTE THIS MIGHT ALSO HAVE GLOBALS IF UNSCALING

    data.edge_attr = (data.edge_attr - distance_min) / distance_range
    data.global_features = (data.global_features - global_min) / global_range
    data.trunk_inputs = (data.trunk_inputs - distance_min) / distance_range
    data.pos = (data.pos - distance_min) / distance_range
    return data


def pre_process(
    data_path,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    scaling_method="run4",
    original_trunk_shape=None,
):
    """
    Preprocess the dataset by computing statistics and scaling features.

    Only supports run4 scaling method (min-max with zero minimums and shared stats).
    Deprecated methods (run1, run2, run3) are no longer supported.

    :param data_path: Path to the dataset
    :param train_size: Fraction or number of samples for training
    :param val_size: Fraction or number of samples for validation
    :param test_size: Fraction or number of samples for testing
    :param scaling_method: Scaling method ('run4' is the only supported method)
    :param original_trunk_shape: Original shape of trunk inputs (n_probes, 2)
    :return: Preprocessed dataset
    """
    # Validate scaling method
    if scaling_method != "run4":
        raise ValueError(
            f"Only 'run4' scaling method is supported. Got: '{scaling_method}'. "
            "Deprecated methods (run1, run2, run3) are no longer available."
        )

    # Load dataset
    dataset = Torch_Geomtric_Dataset(data_path, in_mem=False)

    no_samples = len(dataset)
    no_zips = len(dataset.zip_list)

    no_cases_per_zip = [len(zip_content) - 1 for zip_content in dataset.zip_matrix]
    no_cases_per_zip = np.unique(no_cases_per_zip)
    assert len(no_cases_per_zip) == 1, "Cases per zip is not the same for all zip files"

    # these numbers are flowcases not zip files currently there are 10 flowcases pr. zip file
    train_size = int(np.floor(no_zips * train_size) * no_cases_per_zip)
    val_size = int(np.floor(no_zips * val_size) * no_cases_per_zip)
    test_size = int(np.floor(no_zips * test_size) * no_cases_per_zip)
    assert train_size + val_size + test_size <= no_samples

    print(
        f"Dataset has {no_samples} samples, {no_zips} zip files, {no_cases_per_zip[0]} cases per zip file"
    )

    # Try to load pre-computed split indices from metadata (new format)
    split_indices = load_split_indices_from_metadata(data_path)

    # Build a mapping from layout index to zip file
    # Zip files are named _layout{idx}.zip, so we parse the index from filenames
    def get_layout_idx_from_zip(zip_tuple):
        """Extract layout index from zip path like '/path/_layout123.zip'."""
        zip_path = zip_tuple[0]
        basename = os.path.basename(zip_path)
        # Parse _layout{N}.zip
        return int(basename.replace("_layout", "").replace(".zip", ""))

    # Create mapping: layout_idx -> zip_tuple
    zip_by_layout_idx = {get_layout_idx_from_zip(z): z for z in dataset.zip_matrix}

    if split_indices is not None:
        # Use pre-computed split indices (new format)
        logger.info("Using pre-computed split indices from layouts_metadata.npz")

        train_zip_matrix = [zip_by_layout_idx[i] for i in split_indices["train_indices"]]
        val_zip_matrix = [zip_by_layout_idx[i] for i in split_indices["val_indices"]]
        test_zip_matrix = [zip_by_layout_idx[i] for i in split_indices["test_indices"]]
    else:
        # Fallback: shuffle on-the-fly (backward compatibility with old metadata)
        logger.info("Fallback: shuffling layouts on-the-fly (old metadata format)")

        zip_matrix_shuffled = dataset.zip_matrix.copy()
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        rng.shuffle(zip_matrix_shuffled)
        logger.info("Shuffled layouts before train/val/test split")

        # Split into training, validation and test sets
        zip_content = dataset.zip_matrix[0]
        zip_content = [x for x in zip_content if x.endswith(".pt")]
        flow_cases_per_zip = len(zip_content)
        train_idxs = train_size // flow_cases_per_zip
        val_idxs = val_size // flow_cases_per_zip
        test_idxs = test_size // flow_cases_per_zip
        assert train_idxs + val_idxs + test_idxs <= len(dataset.zip_matrix)

        train_zip_matrix = zip_matrix_shuffled[:train_idxs]
        val_zip_matrix = zip_matrix_shuffled[train_idxs : train_idxs + val_idxs]
        test_zip_matrix = zip_matrix_shuffled[
            train_idxs + val_idxs : train_idxs + val_idxs + test_idxs
        ]

    logger.info(
        f"Split sizes: {len(train_zip_matrix)} train, "
        f"{len(val_zip_matrix)} val, {len(test_zip_matrix)} test layouts"
    )

    # Get metrics from TRAINING dataset only to avoid data leakage
    # Temporarily update dataset.zip_matrix to only include training data
    original_zip_matrix = dataset.zip_matrix
    dataset.zip_matrix = train_zip_matrix
    dataset.zip_paths_repeated, dataset.zip_contents = dataset._create_indexes()

    dataset_stats = retrieve_dataset_stats(
        dataset, max_indexes=len(dataset) - 1, original_trunk_shape=original_trunk_shape
    )

    # Restore original zip_matrix for any subsequent operations
    dataset.zip_matrix = original_zip_matrix
    dataset.zip_paths_repeated, dataset.zip_contents = dataset._create_indexes()

    # Use run4 scaling: min-max with shared stats for distances and velocities
    logger.info("Using run4 scaling method (min-max with zero minimums and shared stats)")
    scale_stats = obtain_min_max_values(dataset_stats)
    scale_stats["scaling_type"] = "min_max"
    scale_stats["scaling_method"] = scaling_method

    ## Save scale stats to file
    # convert to arrays to list for json serialization
    print("Saving scale stats to file")
    json_scale_stats = scale_stats
    for key in json_scale_stats:
        if isinstance(json_scale_stats[key], dict):
            for sub_key in json_scale_stats[key]:
                if isinstance(json_scale_stats[key][sub_key], torch.Tensor):
                    json_scale_stats[key][sub_key] = json_scale_stats[key][sub_key].tolist()
        else:
            if isinstance(json_scale_stats[key], torch.Tensor):
                json_scale_stats[key] = json_scale_stats[key].tolist()

    with open(os.path.join(data_path, "scale_stats.json"), "w") as f:
        json.dump(json_scale_stats, f, indent=4)

    # create folders for the preprocessed datasets
    train_path = os.path.join(data_path, "train_pre_processed")
    val_path = os.path.join(data_path, "val_pre_processed")
    test_path = os.path.join(data_path, "test_pre_processed")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    def scale_and_copy_dataset(zip_matrix, destination_path):
        """Used to scale the datasets and save them to the destination path"""
        # Scale the datasets
        for zip_file in tqdm(zip_matrix):
            zip_path = zip_file[0]
            zip_items = zip_file[1:]

            scaled_data_objects = []
            zip_content_names = []
            with ZipFile(zip_path, "r") as zf:
                for item in zip_items:
                    with zf.open(item) as f:
                        stream = io.BytesIO(f.read())
                        data = torch.load(stream, weights_only=False)
                        # Apply run4 min-max scaling
                        data = min_max_scale(data, scale_stats)
                        data = append_globals_to_nodes(data)
                        scaled_data_objects.append(data)
                        zip_content_names.append(item)

            # Save the scaled data objects
            save_zip_path = os.path.join(destination_path, os.path.basename(zip_path))
            with (
                tempfile.TemporaryDirectory() as tempdir,
                ZipFile(save_zip_path, "w") as zf,
            ):
                for data, zip_content_name in zip(scaled_data_objects, zip_content_names):
                    graph_temp_path = os.path.join(tempdir, item)
                    torch.save(data, graph_temp_path)

                    zf.write(filename=graph_temp_path, arcname=zip_content_name)

    scale_and_copy_dataset(train_zip_matrix, train_path)
    scale_and_copy_dataset(val_zip_matrix, val_path)
    scale_and_copy_dataset(test_zip_matrix, test_path)


def add_split_to_existing_metadata(
    data_path: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    split_seed: int = 42,
):
    """
    Add split indices to an existing layouts_metadata.npz file.

    This allows migrating existing datasets to the new format with embedded
    split indices, without regenerating the entire dataset.

    Args:
        data_path: Path to the dataset directory
        train_frac: Fraction of data for training (default: 0.6)
        val_frac: Fraction of data for validation (default: 0.2)
        test_frac: Fraction of data for testing (default: 0.2)
        split_seed: Random seed for reproducibility (default: 42)
    """
    from utils.preprocessing_utils import generate_split_indices

    metadata_path = os.path.join(data_path, "layouts_metadata.npz")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load existing metadata
    existing_data = dict(np.load(metadata_path, allow_pickle=True))
    n_layouts = int(existing_data["n_layouts"])

    # Check if split already exists
    if "train_indices" in existing_data:
        logger.warning("Split indices already exist in metadata. Overwriting...")

    # Generate split indices
    split_data = generate_split_indices(
        n_layouts=n_layouts,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=split_seed,
    )

    # Add split info to existing data
    existing_data["shuffled_indices"] = split_data["shuffled_indices"]
    existing_data["train_indices"] = split_data["train_indices"]
    existing_data["val_indices"] = split_data["val_indices"]
    existing_data["test_indices"] = split_data["test_indices"]
    existing_data["split_seed"] = split_seed
    existing_data["train_frac"] = train_frac
    existing_data["val_frac"] = val_frac
    existing_data["test_frac"] = test_frac

    # Save updated metadata
    np.savez(metadata_path, **existing_data)

    logger.info(f"Added split indices to: {metadata_path}")
    logger.info(
        f"Split: {len(split_data['train_indices'])} train, "
        f"{len(split_data['val_indices'])} val, {len(split_data['test_indices'])} test"
    )

    # Also save human-readable split info JSON
    types = existing_data["types"].tolist()
    spacings = existing_data["spacings"].tolist()
    n_turbines = existing_data["n_turbines"].tolist()

    def build_layout_info(idx):
        """Build layout info dict for a given index."""
        return {
            "layout_idx": int(idx),
            "type": types[idx],
            "spacing": float(spacings[idx]),
            "n_turbines": int(n_turbines[idx]),
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
            "total_layouts": n_layouts,
            "n_train": len(split_data["train_indices"]),
            "n_val": len(split_data["val_indices"]),
            "n_test": len(split_data["test_indices"]),
        },
        "train": [build_layout_info(i) for i in split_data["train_indices"]],
        "val": [build_layout_info(i) for i in split_data["val_indices"]],
        "test": [build_layout_info(i) for i in split_data["test_indices"]],
    }

    split_info_path = os.path.join(data_path, "split_info.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Split info saved to: {split_info_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess dataset or add split indices to existing metadata"
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default="/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--add-split-only",
        action="store_true",
        help="Only add split indices to existing layouts_metadata.npz (no preprocessing)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="Fraction of data for training (default: 0.6)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)

    if args.add_split_only:
        # Migration mode: only add split indices to existing metadata
        add_split_to_existing_metadata(
            data_path,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            split_seed=args.split_seed,
        )
    else:
        # Full preprocessing mode
        pre_process(
            data_path,
            train_size=args.train_frac,
            val_size=args.val_frac,
            test_size=args.test_frac,
            scaling_method="run4",
        )

        train_path = os.path.join(data_path, "train_pre_processed")
        Torch_Geomtric_Dataset(train_path)
