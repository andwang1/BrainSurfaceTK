import os
import pickle
import pyvista as pv

import numpy as np
import torch
import torch.nn.functional as F
# from deepl_brain_surfaces.shapenet_fake import ShapeNet
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
# Metrics

from ..src.utils import get_comment, get_data_path, data, get_grid_search_local_features

# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
num_points_dict = {'original': 32492, '50': 16247}


def get_file_path(patient_id, session_id):
    file_name = "sub-" + patient_id + "_ses-" + session_id + files_ending
    file_path = data_folder + '/' + file_name
    return file_path


if __name__ == '__main__':

    num_workers = 2
    local_features = []
    grid_features = get_grid_search_local_features(local_features)

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = True

    data_nativeness = 'aligned'
    data_compression = "50"
    data_type = "inflated"

    additional_comment = ''

    experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{additional_comment}'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    lr = 0.001
    batch_size = 8
    global_features = []
    target_class = 'gender'
    task = 'segmentation'
    REPROCESS = False



    # 2. Get the data splits indices
    with open('src/names.pk', 'rb') as f:
        indices = pickle.load(f)

    data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere='left')

    all_faces = []
    for patient_idx in indices['Val']:

        # Get file path to .vtk/.vtp for one patient #TODO: Maybe do something more beautiful
        file_path = get_file_path(patient_idx[:11], patient_idx[12:])
        # If file exists
        if os.path.isfile(file_path):

            print('Getting')
            mesh = pv.read(file_path)

            # Get points
            points = torch.tensor(mesh.points)

            # Get faces
            n_faces = mesh.n_cells
            faces = mesh.faces.reshape((n_faces, -1))

            all_faces.append(faces)


    with open('all_faces.pkl', 'wb') as file:
        pickle.dump(all_faces, file)



























