import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
import vtk
from django.conf import settings
from torch_geometric.data import Data
from vtk.util.numpy_support import vtk_to_numpy

from .pre_trained_models.pointnet2_regression import Net  # TODO

if settings.DEBUG == True:
    MODEL_PATH = os.path.join(os.getcwd(), "GUI/backend/pre_trained_models/model_best.pt")
else:
    MODEL_PATH = os.path.join(os.getcwd(), "backend/pre_trained_models/model_best.pt")


def get_features(list_features, reader):
    '''Returns tensor of features to add in every point.
    :param list_features: list of features to add. Mapping is in self.feature_arrays
    :param mesh: pyvista mesh from which to get the arrays.
    :returns: tensor of features or None if list is empty.'''

    # Very ugly workaround about some classes not being in some data.
    list_of_drawem_labels = [0, 5, 7, 9, 11, 13, 15, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}

    if list_features:

        if 'drawem' in list_features:

            one_hot_drawem = pd.get_dummies(vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(feature_arrays['drawem'])))
            # one_hot_drawem = pd.get_dummies(mesh.get_array(feature_arrays['drawem']))

            new_df = pd.DataFrame()
            for label in list_of_drawem_labels:
                if label not in one_hot_drawem.columns:
                    new_df[label] = 0
                else:
                    new_df[label] = one_hot_drawem[label]

            one_hot_drawem = new_df.to_numpy()

            drawem_list = [one_hot_drawem[:, i] for i in range(one_hot_drawem.shape[1])]

        else:
            drawem_list = []

        # features = [mesh.get_array(feature_arrays[key]) for key in feature_arrays if key != 'drawem']
        features = [vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(feature_arrays[key])) for key in feature_arrays if key != 'drawem']

        return torch.tensor(features + drawem_list).t()
    else:
        return None


def predict_age(file_path="/media/original/data/vtps/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp"):
    torch.manual_seed(0)
    if osp.isfile(file_path):

        # mesh = read(file_path)
        # reader = vtk.vtkPolyDataReader()
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        # output = reader.GetOutput()

        points = torch.tensor(np.array(reader.GetOutput().GetPoints().GetData()))

        local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
        x = get_features(local_features, reader)

        data = Data(batch=torch.zeros_like(x[:, 0]).long(), x=x, pos=points)
        # data = Data(x=x, pos=points)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        numb_local_features = x.size(1)
        numb_global_features = 0

        model = Net(numb_local_features, numb_global_features).to(device)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # data_loader = DataLoader([data], batch_size=1, shuffle=False)
        # print(len(data_loader))
        # pred = model(next(iter(data_loader)).to(device))
        pred = model(data.to(device))

        return pred.item()
    else:
        return 'Unable to predict..'


if __name__ == '__main__':
    print(predict_age(
        '/mnt/UHDD/Programming/Projects/GroupProject/deepl_brain_surfaces/GUI' + "/media/original/data/vtps/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp"))
