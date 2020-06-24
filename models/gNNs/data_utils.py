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

    def __init__(self, files_path, meta_data_filepath, save_path=None, dataset="train", train_split_per=0.8,
                 max_workers=6, save_dataset=False,
                 load_from_pk=True):
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
        # Samples & respective targets
        self.sample_filepaths = None
        self.train_split_per = train_split_per
        self.dataset = dataset
        # Loading
        if load_from_pk and not isinstance(save_path, str):
            raise Exception("Specify a save path!")
        if save_dataset and (save_path is None):
            raise Exception("You want to save the dataset but you haven't specified a save_path")

        if load_from_pk and os.path.exists(self.update_save_path(save_path, dataset)):
            print("Prepared dataset already exists in: ", save_path)
            if dataset == "none":
                train_sample_filepaths = self.get_sample_file_paths(self.update_save_path(save_path, "train"))
                test_sample_filepaths = self.get_sample_file_paths(self.update_save_path(save_path, "test"))
                self.sample_filepaths = train_sample_filepaths + test_sample_filepaths
            else:
                save_path = self.update_save_path(save_path, dataset)
                self.sample_filepaths = self.get_sample_file_paths(save_path)
        else:
            print("Generating Dataset")
            self.generate_dataset(files_path, meta_data_filepath, save_path)

            if dataset == "none":
                train_sample_filepaths = self.get_sample_file_paths(self.update_save_path(save_path, "train"))
                test_sample_filepaths = self.get_sample_file_paths(self.update_save_path(save_path, "test"))
                self.sample_filepaths = train_sample_filepaths + test_sample_filepaths
            else:
                save_path = self.update_save_path(save_path, dataset)
                self.sample_filepaths = self.get_sample_file_paths(save_path)

        print("Initialisation complete")

    def generate_dataset(self, files_path, meta_data_filepath, save_path):

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
                    cycle((train_save_path,))
                ), total=len(train_fps))]
                te_results = [r for r in tqdm(executor.map(
                    self.process_file_target,
                    test_fps,
                    test_targets,
                    cycle((test_save_path,))
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

        self.normalise_dataset(train_save_path)
        self.normalise_dataset(test_save_path)

    def process_file_target(self, file_to_load, target, save_path):

        fp_save_path = os.path.join(save_path, os.path.basename(file_to_load).replace(".vtp", ".pickle"))
        if os.path.exists(fp_save_path):
            return 2

        mesh = pv.read(file_to_load)
        src, dst = zip(*self.convert_face_array_to_edge_array(self.build_face_array(list(mesh.faces))))
        src = np.array(src)
        dst = np.array(dst)
        # Edges are directional in DGL; Make them bi-directional.
        g = dgl.DGLGraph(
            (torch.from_numpy(np.concatenate([src, dst])), torch.from_numpy(np.concatenate([dst, src])))
        )
        features = list()
        for name in mesh.array_names:
            if name in ['corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi']:
                features.append(mesh.get_array(name=name, preference="point"))
        g.ndata['features'] = torch.tensor(np.column_stack(features)).float()
        g.add_edges(g.nodes(), g.nodes())  # Required Trick --> see DGL discussions somewhere sorry
        g.edata['features'] = self.get_edge_data(mesh, src, dst, g)

        self._save_data_with_pickle(fp_save_path, (g, target))

        return 1

    @staticmethod
    def split_dataset(samples, targets, train_split_per):
        train_size = round(len(samples) * train_split_per)
        train_indices = np.random.choice([i for i in range(len(samples))], size=train_size, replace=False)

        train_samples = [samples[i] for i in range(len(samples)) if i in train_indices]
        train_targets = [targets[i] for i in range(len(targets)) if i in train_indices]

        test_samples = [samples[i] for i in range(len(samples)) if i not in train_indices]
        test_targets = [targets[i] for i in range(len(targets)) if i not in train_indices]

        return (train_samples, train_targets), (test_samples, test_targets)

    @staticmethod
    def search_for_files_and_targets(load_path, meta_data_file_path):
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
        if dataset == "none":
            return save_path
        else:
            return os.path.join(save_path, dataset)

    def __len__(self):
        return len(self.sample_filepaths)

    def __getitem__(self, item):
        return self.load_sample_from_pickle(self.sample_filepaths[item])

    @staticmethod
    def build_face_array(face_indices, include_n_verts=False):
        face_arrays = list()
        while len(face_indices) > 0:
            n_verts = face_indices.pop(0)
            face_arrays.append([n_verts] + [face_indices.pop(0) for _ in range(n_verts)])
        if not include_n_verts:
            face_arrays = [lst[1:] for lst in face_arrays]
        return face_arrays

    @staticmethod
    def convert_face_array_to_edge_array(face_array):
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
        files_to_load = [os.path.join(data_path, file_to_load) for file_to_load in os.listdir(data_path) if
                         file_to_load.endswith(".pickle")]
        self.normalise_nodes(files_to_load)
        self.normalise_edges(files_to_load)
        self.normalise_targets(files_to_load)

    def normalise_nodes(self, files_to_load):
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

    def normalise_edges(self, files_to_load):
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

    def normalise_targets(self, files_to_load):
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
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise ZeroDivisionError(f"Normalising went wrong, contains nans or infs")

    def _save_data_with_pickle(self, filepath, data):
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_sample_from_pickle(self, filepath):
        with open(filepath, "rb") as f:
            contents = pickle.load(f)
        return contents

    def get_sample_file_paths(self, ds_store_fp):
        return [os.path.join(ds_store_fp, fp) for fp in os.listdir(ds_store_fp)]

    def load_saved_dataset_with_pickle(self, ds_store_fp):
        if os.path.exists(ds_store_fp):
            fps = [fp for fp in os.listdir(ds_store_fp)]
            data = list()
            for fp in tqdm(fps):
                data.append(self.load_sample_from_pickle(os.path.join(ds_store_fp, fp)))
            return data
        else:
            raise FileNotFoundError("No pickle file exists!")


if __name__ == "__main__":
    # Local
    load_path = os.path.join(os.getcwd(), "data")
    meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    save_path = os.path.join(os.getcwd(), "tmp", "dataset")

    # # Imperial
    # load_path = os.path.join(os.getcwd(), "models", "gNNs", "data")
    # meta_data_file_path = os.path.join(os.getcwd(), "models", "gNNs", "meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "models", "gNNs", "tmp", "dataset")

    dataset = BrainNetworkDataset(load_path, meta_data_file_path, max_workers=0, save_path=save_path, save_dataset=True,
                                  train_split_per=0.5)

    print(dataset[0])

    # test_vtp =  "/mnt/UHDD/Programming/Projects/DeepLearningOnBrains/models/gNNs/tmp_data/vtks"
