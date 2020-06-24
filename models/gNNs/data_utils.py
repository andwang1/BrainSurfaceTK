import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import cycle

import dgl
import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class BrainNetworkDataset(Dataset):
    """
    Dataset for Brain Networks
    """

    def __init__(self, files_path, meta_data_filepath, save_path, dataset="train", train_split_per=0.8, max_workers=8):
        """
        :param files_path: Location of the vtps which will be converted to graphs
        :param meta_data_filepath: filepath of meta_data.tsv (tsv file that contains session/patient ids & scan age
        :param save_path: location for the training and testing data to be saved
        :param dataset: "train", "test", or "none" means all data will be available to the user
        :param train_split_per: ie: 0.8 yields 80% of data to be in the training set and 20% to be in the test dataset
        :param max_workers: number of processes to be used to create the dataset, more is generally faster but don't go
                            higher than 8.
        """
        if not os.path.isdir(files_path):
            raise IsADirectoryError(f"This Location: {files_path} doesn't exist")
        if not os.path.isfile(meta_data_filepath):
            raise FileNotFoundError(f"The meta_data.tsv file address ({meta_data_filepath}) doesn't exist!")
        if dataset not in ["train", "test", "none"]:
            raise ValueError("dataset must be one of: 'train', 'test', 'none'")
        print("Initialising Dataset")
        # Datapaths
        self.path = files_path
        self.meta_data_path = meta_data_filepath
        # Number of workers for loading
        self.max_workers = max_workers

        # Filepaths containing stored graphs and their respective targets
        self.sample_filepaths = None
        self.train_split_per = train_split_per
        self.dataset = dataset
        # Loading
        if os.path.exists(self.update_save_path(save_path, dataset)):
            print("Prepared dataset already exists in: ", save_path)
        else:
            print("Generating Dataset")
            self.generate_dataset(files_path, meta_data_filepath, save_path)

        # Now collect all required filepaths containing data which will be fed to the GNN
        self.sample_filepaths = self.get_sample_file_paths(save_path, dataset)

        print("Initialisation complete")

    def generate_dataset(self, files_path, meta_data_filepath, save_path):
        """
        Generates & saves the dataset to be used by the GNN, NOTE: if any files exist in the train/test folders
        then that file will be skipped!
        :param files_path: Location of the vtps which will be converted to graphs
        :param meta_data_filepath: filepath of meta_data.tsv (tsv file that contains session/patient ids & scan age
        :param save_path: location for the training and testing data to be saved
        :return: None
        """

        # Find files in Imperial Folder & Corresponding targets
        files_to_load, targets = self.search_for_files_and_targets(files_path, meta_data_filepath)

        # Split the dataset here
        (train_fps, train_targets), (test_fps, test_targets) = self.split_dataset(files_to_load,
                                                                                  targets,
                                                                                  self.train_split_per)

        # Make dirs to store data
        train_save_path = self.update_save_path(save_path, "train")
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)
        test_save_path = self.update_save_path(save_path, "test")
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)

        # Convert each mesh to a graph and save
        if self.max_workers > 0:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                tr_results = [r for r in tqdm(executor.map(
                    self.process_file_target,
                    train_fps,
                    train_targets,
                    cycle((train_save_path,)),
                    chunksize=32
                ), total=len(train_fps))]
                te_results = [r for r in tqdm(executor.map(
                    self.process_file_target,
                    test_fps,
                    test_targets,
                    cycle((test_save_path,)),
                    chunksize=32
                ), total=len(test_fps))]
        else:
            tr_results = [r for r in tqdm(map(
                self.process_file_target,
                train_fps,
                train_targets,
                cycle((train_save_path,))
            ), total=len(train_fps))]
            te_results = [r for r in tqdm(map(
                self.process_file_target,
                test_fps,
                test_targets,
                cycle((test_save_path,))
            ), total=len(test_fps))]

        if None in tr_results or None in te_results:
            print("Error during graph building")

        # Normalise the two datasets separately
        # TODO: save guard for when training_split is 0. or 1. otherwise we'll get an error
        self.normalise_dataset(train_save_path)
        self.normalise_dataset(test_save_path)

    def process_file_target(self, file_to_load, target, save_path):
        """
        Process the mesh in the file_to_load (.vtp) and convert it to a graph before pickling (graph, target)
        :param file_to_load: mesh in the file_to_load (.vtp) and convert it to a graph
        :param target: tensor float
        :param save_path: directory for the processed sample to be saved
        :return: 1 meaning success
        """

        fp_save_path = os.path.join(save_path, os.path.basename(file_to_load).replace(".vtp", ".pickle"))
        if os.path.exists(fp_save_path):
            # Response of 2 means this file already exists
            return 2

        mesh = pv.read(file_to_load)
        # Get the edge sources and destinations
        src, dst = zip(*self.convert_face_array_to_edge_array(self.build_face_array(list(mesh.faces))))
        src = np.array(src)
        dst = np.array(dst)
        # Edges are directional in DGL; Make them bi-directional.
        g = dgl.DGLGraph(
            (torch.from_numpy(np.concatenate([src, dst])), torch.from_numpy(np.concatenate([dst, src])))
        )

        g.ndata['features'] = self.get_node_features(mesh)
        g.add_edges(g.nodes(), g.nodes())  # Required Trick --> see DGL discussions somewhere sorry
        g.edata['features'] = self.get_edge_data(mesh, src, dst, g)

        self._save_data_with_pickle(fp_save_path, (g, target))

        return 1

    @staticmethod
    def get_node_features(mesh):
        """
        Extract the point features from the mesh
        :param mesh: pv.PolyData object
        :return: torch tensor float containing features
        """
        features = list()
        for name in mesh.array_names:
            if name in ['corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi']:
                features.append(mesh.get_array(name=name, preference="point"))
        return torch.tensor(np.column_stack(features)).float()

    @staticmethod
    def split_dataset(samples, targets, train_split_per):
        """
        Randomly splits the dataset with replace=False
        :param samples:
        :param targets:
        :param train_split_per:
        :return: (train_samples, train_targets), (test_samples, test_targets)
        """
        train_size = round(len(samples) * train_split_per)
        train_indices = np.random.choice([i for i in range(len(samples))], size=train_size, replace=False)

        train_samples = [samples[i] for i in range(len(samples)) if i in train_indices]
        train_targets = [targets[i] for i in range(len(targets)) if i in train_indices]

        test_samples = [samples[i] for i in range(len(samples)) if i not in train_indices]
        test_targets = [targets[i] for i in range(len(targets)) if i not in train_indices]

        return (train_samples, train_targets), (test_samples, test_targets)

    @staticmethod
    def search_for_files_and_targets(load_path, meta_data_file_path):
        """
        Function to search for potential vtps to be used. The meta_data dataframe is used to select the corresponding
        file_paths and associate them with the respective scan_age.
        :param load_path:
        :param meta_data_file_path:
        :return:
        """
        targets = list()
        df = pd.read_csv(meta_data_file_path, sep='\t', header=0)
        potential_files = [f for f in os.listdir(load_path)]
        files_to_load = list()
        for fn in potential_files:
            participant_id, session_id = fn.split("_")[:2]
            records = df[(df.participant_id == participant_id) & (df.session_id == int(session_id))]
            if len(records) == 1:
                files_to_load.append(os.path.join(load_path, fn))
                targets.append(torch.tensor(records.scan_age.values, dtype=torch.float))
        return files_to_load, targets

    @staticmethod
    def update_save_path(save_path, dataset):
        """
        Update save path with the direction to the training folder or testing folder respectively.
        :param save_path:
        :param dataset:
        :return:
        """
        if dataset == "none":
            return save_path
        else:
            return os.path.join(save_path, dataset)

    def __len__(self):
        return len(self.sample_filepaths)

    def __getitem__(self, item):
        """
        Loads the data from the chosen sample filepath
        :param item: int index
        :return: (graph, target)
        """
        return self.load_sample_from_pickle(self.sample_filepaths[item])

    @staticmethod
    def build_face_array(face_indices, include_n_verts=False):
        """
        Method that makes no assumptions on the mesh being triangulated
        :param face_indices:
        :param include_n_verts: whether or not to include the number of vertices required to build each face
                                in the return
        :return: list containing face_arrays
        """
        face_arrays = list()
        while len(face_indices) > 0:
            n_verts = face_indices.pop(0)
            face_arrays.append([n_verts] + [face_indices.pop(0) for _ in range(n_verts)])
        if not include_n_verts:
            face_arrays = [lst[1:] for lst in face_arrays]
        return face_arrays

    @staticmethod
    def convert_face_array_to_edge_array(face_array):
        """
        Use the face array to return the edge array which is used to build the graph
        :param face_array: output of self.build_face_array
        :return: edges_list
        """
        edges_list = list()
        while len(face_array) > 0:
            face = face_array.pop()
            point = face.pop()
            while len(face) > 0:
                next_point = face.pop()
                edges_list.append((point, next_point))
                point = next_point
        return edges_list

    @staticmethod
    def get_edge_data(mesh, src, dst, g):
        """
        Extract the edge lengths from the mesh to be used by the graph
        :param mesh: pv.PolyData object
        :param src:
        :param dst:
        :param g: graph
        :return: raw edge_lengths
        """
        # TODO: Clean
        edge_lengths = torch.from_numpy(
            np.row_stack([np.sqrt(np.abs(mesh.points[s] ** 2 - mesh.points[d] ** 2).sum())
                          for s, d in zip(src, dst)])
        )
        unnorm_edge_lengths = torch.cat([edge_lengths,
                                         edge_lengths,
                                         torch.zeros(len(g.nodes), 1, dtype=torch.float)])
        return unnorm_edge_lengths

    def normalise_dataset(self, data_path):
        """
        Normalise all features over the dataset!
        :param data_path:
        :return:
        """
        files_to_load = [os.path.join(data_path, file_to_load) for file_to_load in os.listdir(data_path) if
                         file_to_load.endswith(".pickle")]
        # TODO: add safeguard for when split percentage is 0. or 1.
        self.normalise_nodes_(files_to_load)
        self.normalise_edges_(files_to_load)
        self.normalise_targets_(files_to_load)

    def normalise_nodes_(self, files_to_load):
        """
        Normalise the node features over the distribution of all nodes in all graphs available in files_to_load
        :param files_to_load: files_to_load where each pickle file looks like: (graph, target)
        :return: 1 meaning success
        """
        # INPLACE OPERATION

        graph, _ = self.load_sample_from_pickle(files_to_load[0])
        features = graph.ndata["features"]
        mu = torch.tensor(features).sum(dim=0)
        total = len(features)

        for i in range(1, len(files_to_load)):
            graph, _ = self.load_sample_from_pickle(files_to_load[i])
            features = graph.ndata["features"]
            mu += features.sum(dim=0)
            total += len(features)
        mu /= total

        var = 0
        for i in range(0, len(files_to_load)):
            graph, _ = self.load_sample_from_pickle(files_to_load[i])
            features = graph.ndata["features"]
            var += ((features - mu) ** 2).sum(dim=0)
        std = torch.sqrt(var / (total - 1))

        for file_to_load in files_to_load:
            graph, target = self.load_sample_from_pickle(file_to_load)
            graph.ndata["features"] -= mu
            graph.ndata["features"] /= std
            self.check_tensor(graph.ndata["features"])
            self._save_data_with_pickle(file_to_load, (graph, target))

        return 1

    def normalise_edges_(self, files_to_load):
        """
        Normalise the edge features over the distribution of all nodes in all graphs available in files_to_load
        :param files_to_load: files_to_load where each pickle file looks like: (graph, target)
        :return: 1 meaning success
        """

        # INPLACE OPERATION

        graph, _ = self.load_sample_from_pickle(files_to_load[0])
        features = graph.edata["features"]
        mu = torch.tensor(features).sum(dim=0)
        total = len(features)

        for i in range(1, len(files_to_load)):
            graph, _ = self.load_sample_from_pickle(files_to_load[i])
            features = graph.edata["features"]
            mu += features.sum(dim=0)
            total += len(features)
        mu /= total

        var = 0
        for i in range(0, len(files_to_load)):
            graph, _ = self.load_sample_from_pickle(files_to_load[i])
            features = graph.edata["features"]
            var += ((features - mu) ** 2).sum(dim=0)
        std = torch.sqrt(var / (total - 1))

        for file_to_load in files_to_load:
            graph, target = self.load_sample_from_pickle(file_to_load)
            graph.edata["features"] -= mu
            graph.edata["features"] /= std
            self.check_tensor(graph.ndata["features"])
            self._save_data_with_pickle(file_to_load, (graph, target))

        return 1

    def normalise_targets_(self, files_to_load):
        """
        Normalise the targets over the distribution of all nodes in all graphs available in files_to_load
        :param files_to_load: files_to_load where each pickle file looks like: (graph, target)
        :return: 1 meaning success
        """

        # INPLACE OPERATION
        graph, target = self.load_sample_from_pickle(files_to_load[0])
        mu = torch.tensor(target)
        total = 1

        for i in range(1, len(files_to_load)):
            _, target = self.load_sample_from_pickle(files_to_load[i])
            mu += target
            total += 1
        mu /= total

        var = 0
        for i in range(0, len(files_to_load)):
            _, target = self.load_sample_from_pickle(files_to_load[i])
            var += (target - mu) ** 2
        std = torch.sqrt(var / (total - 1))

        for file_to_load in files_to_load:
            graph, target = self.load_sample_from_pickle(file_to_load)
            target -= mu
            target /= std
            self.check_tensor(graph.ndata["features"])
            self._save_data_with_pickle(file_to_load, (graph, target))

        return 1

    @staticmethod
    def check_tensor(tensor):
        """
        Check if the tensor contains any nan or inf values. If so it will raise an error
        :param tensor: tensor to be checked
        :return:
        """
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise ZeroDivisionError(f"Normalising went wrong, contains nans or infs")

    def _save_data_with_pickle(self, filepath, data):
        """
        Pickle dump the data into a file given the filepath
        :param filepath: filepath where the data will be dumped (endswith ".pickle")
        :param data: data to be pickled, typically of form (graph, target)
        :return: None
        """
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_sample_from_pickle(self, filepath):
        """
        Loads the (graph, target) given the filepath
        :param filepath: location to load data from
        :return: contents, usually of the form (graph, target)
        """
        with open(filepath, "rb") as f:
            contents = pickle.load(f)
        return contents

    def get_sample_file_paths(self, ds_store_fp, dataset="none"):
        """
        Given the 'dataset' param, locate all available processed filepaths to be used by the dataset
        :param ds_store_fp: parent directory that contains folders 'train' & 'test'
        :param dataset: 'train' 'test' or 'none'
        :return: sample_file_paths
        """
        if dataset == "none":
            tr_path = self.update_save_path(ds_store_fp, "train")
            tr_fps = [os.path.join(tr_path, fp) for fp in os.listdir(tr_path)]
            te_path = self.update_save_path(ds_store_fp, "test")
            te_fps = [os.path.join(te_path, fp) for fp in os.listdir(te_path)]
            return tr_fps + te_fps
        else:
            parent_path = self.update_save_path(ds_store_fp, dataset)
            return [os.path.join(parent_path, fp) for fp in os.listdir(parent_path)]


if __name__ == "__main__":
    # Local
    load_path = os.path.join(os.getcwd(), "data")
    meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    save_path = os.path.join(os.getcwd(), "tmp", "dataset")

    # # Imperial
    # load_path = os.path.join(os.getcwd(), "models", "gNNs", "data")
    # meta_data_file_path = os.path.join(os.getcwd(), "models", "gNNs", "meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "models", "gNNs", "tmp", "dataset")

    dataset = BrainNetworkDataset(load_path, meta_data_file_path, max_workers=0,
                                  save_path=save_path, train_split_per=0.5, dataset="train")

    print(dataset[0])
