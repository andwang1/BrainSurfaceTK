import os
import pickle
from concurrent.futures import ProcessPoolExecutor

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

    def __init__(self, files_path, meta_data_filepath, save_path=None, max_workers=6, save_dataset=False,
                 load_from_pk=True):
        if not os.path.isdir(files_path):
            raise IsADirectoryError(f"This Location: {files_path} doesn't exist")
        if not os.path.isfile(meta_data_filepath):
            raise FileNotFoundError(f"The meta_data.tsv file address ({meta_data_filepath}) doesn't exist!")
        print("Initialising Dataset")
        # Datapaths
        self.path = files_path
        self.meta_data_path = meta_data_filepath
        # Number of workers for loading
        self.max_workers = max_workers
        # Samples & respective targets
        self.sample_filepaths = None
        # Loading
        if load_from_pk and isinstance(save_path, str):
            if os.path.exists(save_path):
                print("Prepared dataset already exists in: ", save_path)
                self.sample_filepaths = self.get_sample_file_paths(save_path)
            else:
                print("Generating Dataset")
                self.load_dataset(files_path, meta_data_filepath)
                self.convert_ds_to_tensors()
                if save_dataset and (save_path is not None):
                    print("Saving")
                    self.save_dataset_with_pickle(save_path)
        else:
            print("Generating Dataset")
            self.load_dataset(files_path, meta_data_filepath)
            self.convert_ds_to_tensors()
            if save_dataset and (save_path is not None):
                print("Saving")
                self.save_dataset_with_pickle(save_path)
        print("Initialisation complete")

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

    def load_dataset(self, load_path, meta_data_file_path):
        self.samples = list()
        self.targets = list()
        df = pd.read_csv(meta_data_file_path, sep='\t', header=0)
        potential_files = [f for f in os.listdir(load_path)]
        files_to_load = list()
        for fn in potential_files:
            participant_id, session_id = fn.split("_")[:2]
            records = df[(df.participant_id == participant_id) & (df.session_id == int(session_id))]
            if len(records) == 1:
                files_to_load.append(os.path.join(load_path, fn))
                self.targets.append(records.scan_age.values)
        del df
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.samples = [graph for graph in tqdm(executor.map(self.load_data, files_to_load),
                                                    total=len(files_to_load))]

            # graph = self.samples[0]
            # features = graph.ndata["features"]
            # mu = torch.tensor(features).sum(dim=0)
            # total = len(features)
            # for i in range(1, len(self.samples)):
            #     graph = self.samples[i]
            #     features = graph.ndata["features"]#.copy()
            #     mu += features.sum(dim=0)
            #     total += len(features)
            # mu /= total
            # var = 0
            # for i in range(0, len(self.samples)):
            #     graph = self.samples[i]
            #     features = graph.ndata["features"]#.copy()
            #     var += ((features - mu) ** 2).sum(dim=0)
            # std = torch.sqrt(var / (total - 1))
            #
            # # all_features = np.row_stack([graph.ndata["features"] for graph in self.samples])
            # # test_mu = np.mean(all_features, axis=0, keepdims=True)
            # # test_std = np.std(all_features, axis=0, ddof=1, keepdims=True)
            # # print(torch.all(mu == torch.from_numpy(test_mu)))
            # # print(torch.all(std == torch.from_numpy(test_std)))
            #
            # for graph in self.samples:
            #     graph.ndata["features"] -= mu
            #     graph.ndata["features"] /= std
        return self.samples, self.targets

    def convert_ds_to_tensors(self):
        # Don't believe self.samples needs to be converted.
        # self.targets = np.array(self.targets, dtype=np.float)
        self.targets = torch.tensor(self.targets, dtype=torch.float).view((-1, 1))
        self.targets = (self.targets - self.targets.mean()) / self.targets.std()
        print("Targets normalised")

    def load_data(self, filepath):
        mesh = pv.read(filepath)
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
        g.add_edges(g.nodes(), g.nodes())
        return g

    def save_dataset_with_pickle(self, ds_store_fp):
        if not os.path.exists(ds_store_fp):
            os.makedirs(ds_store_fp)
        if self.max_workers > 1:
            filepaths = [os.path.join(ds_store_fp, f"{i}.pickle") for i in range(len(self.samples))]
            with ProcessPoolExecutor(max_workers=self.max_workers) as e:
                [0 for _ in tqdm(e.map(self._save_data_with_pickle, filepaths, zip(*(self.samples, self.targets))),
                                 total=len(self.targets))]
        else:
            for i in tqdm(range(len(self.samples)), total=len(self.samples)):
                pair = (self.samples[i], self.targets[i])
                with open(os.path.join(ds_store_fp, f"{i}.pickle"), "wb") as f:
                    pickle.dump(pair, f)
        return ds_store_fp

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
                with open(os.path.join(ds_store_fp, fp), "rb") as f:
                    data.append(pickle.load(f))
            return zip(*data)
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
    dataset = BrainNetworkDataset(load_path, meta_data_file_path, max_workers=8, save_path=save_path, save_dataset=True)

    # test_vtp =  "/mnt/UHDD/Programming/Projects/DeepLearningOnBrains/models/gNNs/tmp_data/vtks"
