import os
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

    def __init__(self, files_path, meta_data_filepath, max_workers=6):
        if not os.path.isdir(files_path):
            raise IsADirectoryError("This Location doesn't exist")
        if not os.path.isfile(meta_data_filepath):
            raise FileExistsError("The meta_data.tsv file doesn't exist!")
        print("Initialising Dataset")
        # Datapaths
        self.path = files_path
        self.meta_data_path = meta_data_filepath
        # Number of workers for loading
        self.max_workers = max_workers
        # Samples & respective targets
        self.samples = None
        self.targets = None
        # Loading
        self.load_dataset(files_path, meta_data_filepath)
        self.convert_ds_to_tensors()
        print("Initialisation complete")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.samples[item], self.targets[item]

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
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.samples = [graph for graph in tqdm(executor.map(self.load_data, files_to_load),
                                                    total=len(files_to_load))]
        return self.samples, self.targets

    def convert_ds_to_tensors(self):
        # Don't believe self.samples needs to be converted.
        # self.targets = np.array(self.targets, dtype=np.float)
        self.targets = torch.tensor(self.targets, dtype=torch.float).view((-1, 1))

    def load_data(self, filepath):
        mesh = pv.read(filepath)
        point_array = mesh.points.copy()
        src, dst = zip(*self.convert_face_array_to_edge_array(self.build_face_array(list(mesh.faces))))
        src = np.array(src)
        dst = np.array(dst)
        # Edges are directional in DGL; Make them bi-directional.
        G = dgl.DGLGraph(
            (torch.from_numpy(np.concatenate([src, dst])), torch.from_numpy(np.concatenate([dst, src]))))
        G.ndata['feat'] = torch.tensor(point_array)
        return G


if __name__ == "__main__":
    load_path = os.path.join(os.getcwd(), "models", "gNNs", "data")
    meta_data_file_path = os.path.join(os.getcwd(), "models", "gNNs", "meta_data.tsv")
    dataset = BrainNetworkDataset(load_path, meta_data_file_path, max_workers=8)
