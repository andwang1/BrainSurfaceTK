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

    def __init__(self, files_path, meta_data_filepath, save_path, dataset="train", index_split_pickle_fp=None,
                 train_split_per=(0.8, 0.1, 0.1), max_workers=8):
        """
        :param files_path: Location of the vtps which will be converted to graphs
        :param meta_data_filepath: filepath of meta_data.tsv (tsv file that contains session/patient ids & scan age
        :param save_path: location for the training and testing data to be saved
        :param dataset: "train", "test", "val", or "none" means all data will be available to the user
        :param train_split_per: ie: 0.8 yields 80% of data to be in the training set and 20% to be in the test dataset
        :param max_workers: number of processes to be used to create the dataset, more is generally faster but don't go
                            higher than 8.
        """
        if not os.path.isdir(files_path):
            raise IsADirectoryError(f"This Location: {files_path} doesn't exist")
        if not os.path.isfile(meta_data_filepath):
            raise FileNotFoundError(f"The meta_data.tsv file address ({meta_data_filepath}) doesn't exist!")
        if dataset not in ["train", "test", "val", "none"]:
            raise ValueError("dataset must be one of: 'train', 'test', 'val', 'none'")
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
            self.generate_dataset(files_path, meta_data_filepath, save_path, index_split_pickle_fp)

        # Now collect all required filepaths containing data which will be fed to the GNN
        self.sample_filepaths, self.targets_mu, self.targets_std = self.get_sample_file_paths(save_path, dataset)

        print("Initialisation complete")

    def generate_dataset(self, files_path, meta_data_filepath, save_path, index_split_pickle_fp=None):
        """
        Generates & saves the dataset to be used by the GNN, NOTE: if any files exist in the train/test folders
        then that file will be skipped!
        :param files_path: Location of the vtps which will be converted to graphs
        :param meta_data_filepath: filepath of meta_data.tsv (tsv file that contains session/patient ids & scan age
        :param save_path: location for the training and testing data to be saved
        :return: None
        """

        if index_split_pickle_fp is None:
            # Find files in Imperial Folder & Corresponding targets
            files_to_load, targets = self.search_for_files_and_targets(files_path, meta_data_filepath)
            # Split the dataset here
            (train_fps, train_targets), \
            (val_fps, val_targets), \
            (test_fps, test_targets) = self.split_dataset(files_to_load, targets, self.train_split_per)

        else:
            # Predetermined split
            (train_fps, train_targets), \
            (val_fps, val_targets), \
            (test_fps, test_targets) = self.fetch_data_using_manual_split(files_path, meta_data_filepath,
                                                                          index_split_pickle_fp)

        # Make dirs to store data
        train_save_path = self.update_save_path(save_path, "train")
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)
        val_save_path = self.update_save_path(save_path, "val")
        if not os.path.exists(val_save_path):
            os.makedirs(val_save_path)
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
                val_results = [r for r in tqdm(executor.map(
                    self.process_file_target,
                    val_fps,
                    val_targets,
                    cycle((val_save_path,)),
                    chunksize=32
                ), total=len(val_fps))]
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
            val_results = [r for r in tqdm(map(
                self.process_file_target,
                val_fps,
                val_targets,
                cycle((val_save_path,)),
            ), total=len(val_fps))]
            te_results = [r for r in tqdm(map(
                self.process_file_target,
                test_fps,
                test_targets,
                cycle((test_save_path,))
            ), total=len(test_fps))]

        if (None in tr_results) or (None in val_results) or (None in te_results):
            print("Error during graph building")

        # Normalise the two datasets separately
        # TODO: save guard for when training_split is 0. or 1. otherwise we'll get an error
        self.normalise_dataset(train_save_path)
        self.normalise_dataset(val_save_path)
        self.normalise_dataset(test_save_path)

    def process_file_target(self, file_to_load, age, save_path):
        """
        Process the mesh in the file_to_load (.vtp) and convert it to a graph before pickling (graph, target)
        :param file_to_load: mesh in the file_to_load (.vtp) and convert it to a graph
        :param age: tensor float
        :param save_path: directory for the processed sample to be saved
        :return: 1 meaning success
        """

        fp_save_path = os.path.join(save_path, os.path.basename(file_to_load).replace(".vtp", ".pickle"))
        if os.path.exists(fp_save_path):
            # Response of 2 means this file already exists
            return 2

        # Load mesh
        mesh = pv.read(file_to_load)
        # Get the edge sources and destinations
        src, dst = zip(*self.convert_face_array_to_edge_array(self.build_face_array(list(mesh.faces))))
        src = np.array(src)
        dst = np.array(dst)
        # Edges are directional in DGL; Make them bi-directional.
        g = dgl.DGLGraph(
            (torch.from_numpy(np.concatenate([src, dst])), torch.from_numpy(np.concatenate([dst, src])))
        )

        g.ndata['features'], g.ndata['segmentation'] = self.get_node_features(mesh)
        g.add_edges(g.nodes(), g.nodes())  # Required Trick --> see DGL discussions somewhere sorry
        g.edata['features'] = self.get_edge_data(mesh, src, dst, g)

        self._save_data_with_pickle(fp_save_path, (g, age))

        return 1

    @staticmethod
    def get_node_features(mesh):
        """
        Extract the point features from the mesh
        :param mesh: pv.PolyData object
        :return: torch tensor float containing features
        """
        features = [mesh.points]
        segmentation = list()
        for name in mesh.array_names:
            if name in ['corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi']:
                features.append(mesh.get_array(name=name, preference="point"))
            if name == 'segmentation':
                segmentation.append(mesh.get_array(name=name, preference="point"))

        features = torch.tensor(np.column_stack(features)).float()
        segmentation = torch.from_numpy(np.column_stack(segmentation)).long()
        return features, segmentation

    @staticmethod
    def split_dataset(samples, targets, train_val_test_split: tuple = (0.75, 0.1, 0.15)):
        """
        Randomly splits the dataset with replace=False
        :param samples:
        :param targets:
        :param train_split_per:
        :return: (train_samples, train_targets), (test_samples, test_targets)
        """

        train_size = round(len(samples) * train_val_test_split[0])
        val_size = round(len(samples) * train_val_test_split[1])
        test_size = round(len(samples) * train_val_test_split[2])
        if (train_size + val_size + test_size) != len(samples):
            raise Exception("implementation error")

        available_indices = [i for i in range(len(samples))]

        train_indices = np.random.choice(available_indices, size=train_size, replace=False)
        train_samples = [samples[i] for i in range(len(samples)) if i in train_indices]
        train_targets = [targets[i] for i in range(len(targets)) if i in train_indices]

        # Remove used training indices from bag
        available_indices = [a for a in available_indices if a not in train_indices]
        val_indices = np.random.choice(available_indices, size=val_size, replace=False)
        val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]
        val_targets = [targets[i] for i in range(len(targets)) if i in val_indices]

        # Remove indices used in val from bag
        available_indices = [a for a in available_indices if a not in val_indices]
        test_indices = np.random.choice(available_indices, size=test_size, replace=False)  # Not strictly necessary
        test_samples = [samples[i] for i in range(len(samples)) if i in test_indices]
        test_targets = [targets[i] for i in range(len(targets)) if i in test_indices]

        return (train_samples, train_targets), (val_samples, val_targets), (test_samples, test_targets)

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

    def fetch_data_using_manual_split(self, load_path, meta_data_file_path, index_split_pickle_fp):
        # "names_04152020_noCrashSubs.pk"
        with open(index_split_pickle_fp, "rb") as f:
            indices = pickle.load(f)
        train_indices = indices["Train"]
        val_indices = indices["Val"]
        test_indices = indices["Test"]

        train_files, train_targets = self.get_file_paths_using_indices(load_path, meta_data_file_path, train_indices)
        val_files, val_targets = self.get_file_paths_using_indices(load_path, meta_data_file_path, val_indices)
        test_files, test_targets = self.get_file_paths_using_indices(load_path, meta_data_file_path, test_indices)

        return (train_files, train_targets), (val_files, val_targets), (test_files, test_targets)

    @staticmethod
    def get_file_paths_using_indices(load_path, meta_data_file_path, indices):
        # TODO: REWRITE THIS COS INDICES ARENT INDICES THEY ARE JOINT PARTICIPANT SESSION IDS
        targets = list()
        df = pd.read_csv(meta_data_file_path, sep='\t', header=0)
        potential_files = [f for f in os.listdir(load_path)]
        files_to_load = list()
        for fn in potential_files:
            tmp = fn.replace("sub-", "").replace("ses-", "").split("_")[:2]
            if ("_".join(tmp) in indices) and ("left" in fn):
                participant_id, session_id = tmp
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
        targets_mu, targets_std = self.normalise_targets_(files_to_load)
        self._save_data_with_pickle(os.path.join(data_path, "mu_std.pickle"), (targets_mu, targets_std))

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
            self.check_tensor(graph.edata["features"])
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
            self.check_tensor(target)
            self._save_data_with_pickle(file_to_load, (graph, target))

        return mu, std

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
            val_path = self.update_save_path(ds_store_fp, "val")
            val_fps = [os.path.join(tr_path, fp) for fp in os.listdir(val_path)]
            te_path = self.update_save_path(ds_store_fp, "test")
            te_fps = [os.path.join(te_path, fp) for fp in os.listdir(te_path)]
            return tr_fps + val_fps + te_fps
        else:
            parent_path = self.update_save_path(ds_store_fp, dataset)
            targets_mu, targets_std = self.load_sample_from_pickle(os.path.join(parent_path, "mu_std.pickle"))
            return [os.path.join(parent_path, fp) for fp in os.listdir(parent_path) if
                    fp != "mu_std.pickle"], targets_mu, targets_std


if __name__ == "__main__":
    # # Local
    # load_path = os.path.join(os.getcwd(), "data")
    # pickle_split_filepath = os.path.join(os.getcwd(), "names_06152020_noCrashSubs.pk")
    # meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "tmp", "dataset")

    # Imperial
    load_path = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_10k/white/vtk"
    pickle_split_filepath = "/vol/bitbucket/cnw119/neodeepbrain/models/gNNs/names_06152020_noCrashSubs.pk"
    meta_data_file_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    save_path = "/vol/bitbucket/cnw119/tmp/basicdataset"

    dataset = BrainNetworkDataset(load_path, meta_data_file_path, max_workers=0,
                                  save_path=save_path, train_split_per=(0.4, 0.3, 0.3), dataset="train",
                                  index_split_pickle_fp=pickle_split_filepath)

    print(dataset.targets_mu, dataset.targets_std)

    print(dataset[0])
