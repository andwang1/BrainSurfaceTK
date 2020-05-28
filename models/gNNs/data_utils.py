from torch.utils.data.dataset import Dataset
import torch
import dgl
import os
from tqdm import tqdm
import numpy as np
import pyvista as pv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
import time

class BrainNetworkDataset(Dataset):
    """
    Dataset for Brain Networks
    """

    def __init__(self, path, targets=[1 for _ in range(6)], max_workers=6):
        print("Initialising Dataset")
        self.path = path
        self.targets = torch.tensor(targets, dtype=torch.float).view((-1, 1))
        self.max_workers = max_workers
        self.samples = None
        self.load_dataset(path)
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

    def load_dataset(self, load_path):
        self.samples = list()
        filepaths = [os.path.join(load_path, f) for f in os.listdir(load_path)]
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self.samples = [graph for graph in tqdm(executor.map(self.load_data, filepaths), total=len(filepaths))]
        return self.samples

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
    dataset = BrainNetworkDataset(load_path, max_workers=8)

