import pickle
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
import pyvista as pv
import torch

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset

from src.read_meta import read_meta

data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/inflated"
files_ending = "_hemi-L_inflated_reduce50.vtk"
feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}


def get_file_path(patient_id, session_id):
    file_name = "sub-" + patient_id + "_ses-" + session_id + files_ending
    file_path = data_folder + '/' + file_name
    return file_path


with open('src/names.pk', 'rb') as f:
    indices = pickle.load(f)


for mode in ['Train', 'Val', 'Test']:
    print(f'Checking all arrays in {mode}')
    for patient_idx in tqdm(indices[mode]):
        for array in ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']:

            file_path = get_file_path(patient_idx[:11], patient_idx[12:])

            # If file exists
            if os.path.isfile(file_path):
                mesh = pv.read(file_path)

                # IF array is empty
                if mesh.get_array(feature_arrays[array]).size == 0:
                    print('='*30)
                    print('=' * 30)
                    print(array)
                    print(f'{patient_idx}\n\n ', mesh.get_array(feature_arrays[array]))