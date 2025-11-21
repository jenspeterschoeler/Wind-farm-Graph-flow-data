import datetime
import io
import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Tuple, Union
from zipfile import ZipFile

import numba as nb
import numpy as np
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from to_graph import PyGTupleData

NestedListStr = Union[str, List["NestedListStr"]]
logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")


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
                for zip_path, zip_item in zip(
                    self.zip_paths_repeated, self.zip_contents
                )
            ]

    def _open_zip(self, zip_construct: List[str]):
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

    def _open_single_content_in_zip(self, zip_construct: List[str]):
        zip_path = zip_construct[0]
        zip_item = zip_construct[1]
        with ZipFile(zip_path) as zf:
            with zf.open(zip_item) as f:
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

    def _create_indexes(self) -> Tuple[np.ndarray, np.ndarray]:
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
        zip_construct = (self.zip_paths_repeated[idx], self.zip_contents[idx])

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
    def finalize(self) -> Dict:
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
    dataset: Torch_Geomtric_Dataset, max_indexes: int, original_trunk_shape: Tuple
) -> Dict:
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

    osa_node = online_stats_alg((random_graph.node_features.shape[-1]))
    osa_edge = online_stats_alg((random_graph.edge_attr.shape[-1]))
    osa_global = online_stats_alg((random_graph.global_features.shape[-1]))
    osa_trunk = online_stats_alg(random_trunk.shape[-1], vals_2d=True)
    osa_output = online_stats_alg(random_output.shape[-1], vals_2d=True)

    for i, graph in tqdm(enumerate(dataset)):
        trunk = graph.trunk_inputs
        output = graph.output_features
        # Loop to get the max values of the graph
        graph_shapes = {}
        for key, value in graph:
            if key in ["layout_type", "wt_spacing", "n_wt"]:
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
    for key in stats.keys():
        if isinstance(stats[key], dict):
            for sub_key in stats[key].keys():
                if isinstance(stats[key][sub_key], np.ndarray):
                    stats[key][sub_key] = stats[key][sub_key].tolist()
        else:
            if isinstance(stats[key], np.ndarray):
                stats[key] = stats[key].tolist()
    return stats


def retrieve_dataset_stats(
    dataset: Torch_Geomtric_Dataset, max_indexes: int, original_trunk_shape: Tuple
) -> Dict:
    path = dataset._root_path  # Access the private attribute of the dataset
    stats_path = os.path.join(path, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
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


# TODO make this use shared mean and std for things relating to distances
# TODO make the velocities share mean and std for the same reason
def standard_scale(data, stats):
    """
    Standard scale the dataset
    """
    print(
        "This function does not use shared stats for distances and velocities, also it does not scale positions"
    )
    data.node_features = (
        data.node_features - torch.Tensor(stats["node_features"]["mean"])
    ) / torch.Tensor(stats["node_features"]["std"])
    data.edge_attr = (
        data.edge_attr - torch.Tensor(stats["edge_features"]["mean"])
    ) / torch.Tensor(stats["edge_features"]["std"])
    data.global_features = (
        data.global_features - torch.Tensor(stats["global_features"]["mean"])
    ) / torch.Tensor(stats["global_features"]["std"])
    data.trunk_inputs = (
        data.trunk_inputs - torch.Tensor(stats["trunk"]["mean"])
    ) / torch.Tensor(stats["trunk"]["std"])
    data.output_features = (
        data.output_features - torch.Tensor(stats["output"]["mean"])
    ) / torch.Tensor(stats["output"]["std"])
    return data


def obtain_min_max_values(stats, scaling_method="run2"):

    distance_min = torch.Tensor(
        stats["trunk"]["min"]
    ).min()  #! The min and max are shared for all distances, this is not always default but it does maintain the aspect ratio of the distances.
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

    if scaling_method == "run3" or scaling_method == "run4":
        logger.info(
            "This function 'obtain_min_max_values()' uses some hardcoded values for the global values i.e location of u and ti it is assumed u is the first global feature and ti is the second also node features containt ws_eff, ti_eff, CT, and positions are cartesian"
        )
        assert len(stats["node_features"]["min"]) == 3
        ws_eff_min = torch.Tensor([stats["node_features"]["min"][0]])
        ws_eff_max = torch.Tensor([stats["node_features"]["max"][0]])

        ti_eff_min = torch.Tensor([stats["node_features"]["min"][1]])
        ti_eff_max = torch.Tensor([stats["node_features"]["max"][1]])

        ct_min = torch.Tensor([stats["node_features"]["min"][2]])
        ct_range = torch.Tensor([stats["node_features"]["max"][2]]) - ct_min

        velocity_min = torch.min(velocity_min, ws_eff_min)
        velocity_max = torch.max(velocity_max, ws_eff_max)
        ti_min = torch.min(ti_min, ti_eff_min)
        ti_max = torch.max(ti_max, ti_eff_max)

        velocity_range = velocity_max - velocity_min
        ti_range = ti_max - ti_min

    else:
        assert len(stats["node_features"]["min"]) == 1
        logger.info(
            "This function 'obtain_min_max_values()' uses some hardcoded values for the global values i.e location of u and ti it is assumed u is the first global feature and ti is the second also node features only containt CT, and positions are cartesian"
        )
        ct_min = torch.Tensor([stats["node_features"]["min"]])
        ct_range = torch.Tensor([stats["node_features"]["max"]]) - ct_min

    scale_stats = {
        "distance": {"min": distance_min, "range": distance_range},
        "velocity": {"min": velocity_min, "range": velocity_range},
        "ti": {"min": ti_min, "range": ti_range},
        "ct": {"min": ct_min, "range": ct_range},
    }
    return scale_stats


def min_max_scale(data, scale_stats, scaling_method="run2"):
    """Min-max scale the dataset"""
    logger.info(
        "This function uses some hardcoded values for the global values i.e location of u and ti it is assumed u is the first global feature and ti is the second also node features only containt CT, and positions are cartesian"
    )
    velocity_min = torch.tensor(scale_stats["velocity"]["min"])
    velocity_range = torch.tensor(scale_stats["velocity"]["range"])
    distance_min = torch.tensor(scale_stats["distance"]["min"])
    distance_range = torch.tensor(scale_stats["distance"]["range"])
    ti_min = torch.tensor(scale_stats["ti"]["min"])
    ti_range = torch.tensor(scale_stats["ti"]["range"])
    ct_min = torch.tensor(scale_stats["ct"]["min"])
    ct_range = torch.tensor(scale_stats["ct"]["range"])

    if scaling_method == "run3":
        # HACK! zeros to enable res-net if we want
        velocity_min = torch.tensor([0])  #
        ti_min = torch.tensor([0])
        ct_min = torch.tensor([0])
    elif scaling_method == "run4":
        # HACK! zeros to enable res-net if we want, also required for constructing the connections on the fly e.g. for probe setup
        distance_min = torch.tensor([0])
        velocity_min = torch.tensor([0])  #
        ti_min = torch.tensor([0])
        ct_min = torch.tensor([0])

    node_min = torch.cat([velocity_min, ti_min, ct_min], dim=0)
    node_range = torch.cat([velocity_range, ti_range, ct_range], dim=0)
    global_min = torch.cat([velocity_min, ti_min], dim=0)
    global_range = torch.cat([velocity_range, ti_range], dim=0)

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
    scaling_method="run1",
    original_trunk_shape=None,
):
    """
    Preprocess the dataset

    :param data_path: Path to the dataset
    :return: Preprocessed dataset
    """

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
    # Get metrics from dataset
    dataset_stats = retrieve_dataset_stats(
        dataset, max_indexes=train_size - 1, original_trunk_shape=original_trunk_shape
    )
    if scaling_method == "run1":
        """This method had some issues with not using shared stats for distances and velocities, it should not be used but it is here for backwards compatability and to show the difference between the two methods"""
        scale_stats = dataset_stats
        scale_stats["scaling_type"] = "standard"

    elif (
        scaling_method == "run2" or scaling_method == "run3" or scaling_method == "run4"
    ):
        """Here we use shared stats for distances and velocities, this is the recommended method and specifically min max scaling"""
        scale_stats = obtain_min_max_values(
            dataset_stats, scaling_method=scaling_method
        )  # this function setsup the correct values for min and max for the different features dimensions
        scale_stats["scaling_type"] = "min_max"

    else:
        print("Missing implementation saving scaled stats")

    scale_stats["scaling_method"] = scaling_method

    ## Save scale stats to file
    # convert to arrays to list for json serialization
    print("Saving scale stats to file")
    json_scale_stats = scale_stats
    for key in json_scale_stats.keys():
        if isinstance(json_scale_stats[key], dict):
            for sub_key in json_scale_stats[key].keys():
                if isinstance(json_scale_stats[key][sub_key], torch.Tensor):
                    json_scale_stats[key][sub_key] = json_scale_stats[key][
                        sub_key
                    ].tolist()
        else:
            if isinstance(json_scale_stats[key], torch.Tensor):
                json_scale_stats[key] = json_scale_stats[key].tolist()

    with open(os.path.join(data_path, "scale_stats.json"), "w") as f:
        json.dump(json_scale_stats, f, indent=4)

    # Split into training, validation and test sets
    zip_content = dataset.zip_matrix[0]
    zip_content = [x for x in zip_content if x.endswith(".pt")]
    flow_cases_per_zip = len(zip_content)
    train_idxs = train_size // flow_cases_per_zip
    val_idxs = val_size // flow_cases_per_zip
    test_idxs = test_size // flow_cases_per_zip
    assert train_idxs + val_idxs + test_idxs <= len(dataset.zip_matrix)

    train_zip_matrix = dataset.zip_matrix[:train_idxs]
    val_zip_matrix = dataset.zip_matrix[train_idxs : train_idxs + val_idxs]
    test_zip_matrix = dataset.zip_matrix[
        train_idxs + val_idxs : train_idxs + val_idxs + test_idxs
    ]

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
                        if scaling_method == "run1":
                            data = standard_scale(data, dataset_stats)
                        elif (
                            scaling_method == "run2"
                            or scaling_method == "run3"
                            or scaling_method == "run4"
                        ):
                            data = min_max_scale(
                                data, scale_stats, scaling_method=scaling_method
                            )
                        data = append_globals_to_nodes(data)
                        scaled_data_objects.append(data)
                        zip_content_names.append(item)

            # Save the scaled data objects
            save_zip_path = os.path.join(destination_path, os.path.basename(zip_path))
            with tempfile.TemporaryDirectory() as tempdir:
                with ZipFile(save_zip_path, "w") as zf:
                    for data, zip_content_name in zip(
                        scaled_data_objects, zip_content_names
                    ):
                        graph_temp_path = os.path.join(tempdir, item)
                        torch.save(data, graph_temp_path)

                        zf.write(filename=graph_temp_path, arcname=zip_content_name)

    scale_and_copy_dataset(train_zip_matrix, train_path)
    scale_and_copy_dataset(val_zip_matrix, val_path)
    scale_and_copy_dataset(test_zip_matrix, test_path)


if __name__ == "__main__":
    # data_path = "./data/medium_graphs"
    data_path = os.path.abspath(
        "/work/users/jpsch/SPO_sophia_dir/data/large_graphs_nodes_2_v2"
    )

    pre_process(
        data_path,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        scaling_method="run4",
    )

    train_path = os.path.join(data_path, "train_pre_processed")
    train_dataset = Torch_Geomtric_Dataset(train_path)
