import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import vtk
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy
from django.conf import settings
import nvidia_smi
from .pre_trained_models.pointnet2_segmentation import Net


if settings.DEBUG == True:
    MODEL_PATH = os.path.join(os.getcwd(), "GUI/backend/pre_trained_models/best_acc_model.pt")
else:
    MODEL_PATH = os.path.join(os.getcwd(), "backend/pre_trained_models/best_acc_model.pt")

# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
num_points_dict = {'original': 32492, '50': 16247}

recording = True
# Limit of memory in MB when to run on GPU if it's available.
GPU_MEM_LIMIT = 2000


def get_features(list_features, reader):
    '''Returns tensor of features to add in every point.
    :param list_features: list of features to add. Mapping is in self.feature_arrays
    :param mesh: pyvista mesh from which to get the arrays.
    :returns: tensor of features or None if list is empty.'''

    # feature_arrays = ['corrected_thickness', 'curvature', 'sulcal_depth']
    drawem_list = []

    features = [vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(array)) for array in list_features]

    return torch.tensor(features + drawem_list).t()


def get_labels(reader):
    '''Returns tensor of features to add in every point.
    :param list_features: list of features to add. Mapping is in self.feature_arrays
    :param mesh: pyvista mesh from which to get the arrays.
    :returns: tensor of features or None if list is empty.'''

    labels = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray('segmentation'))
    return torch.tensor(labels).t()


def segment(brain_path, folder_path_to_write, tmp_file_name=None):

    # Boolean to check if the specified file has segmentation labels
    labels_exist = True

    if tmp_file_name is None:
        tmp_file_name = 'segmented_brain.vtp'

    torch.manual_seed(0)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(brain_path)
    reader.Update()

    points = torch.tensor(np.array(reader.GetOutput().GetPoints().GetData()))

    local_features = ['corrected_thickness', 'curvature', 'sulcal_depth']
    x = get_features(local_features, reader)

    try:
        y = get_labels(reader)
    except:
        labels_exist = False


    pre_transform = T.NormalizeScale()

    if labels_exist:
        data = Data(batch=torch.zeros_like(x[:, 0]).long(), x=x, y=y, pos=points)
    else:
        data = Data(batch=torch.zeros_like(x[:, 0]).long(), x=x, pos=points)

    data = pre_transform(data)

    try:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_res.free / 1024 ** 2
    except:
        free_mem = 0

    device = torch.device('cuda' if torch.cuda.is_available() and free_mem >= GPU_MEM_LIMIT else 'cpu')

    numb_local_features = x.size(1)
    numb_global_features = 0

    model = Net(18, numb_local_features).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.eval()

    # 1. Get predictions and loss
    data = data.to(device)
    out = model(data)

    # 2. Get d (positions), _y (actual labels), _out (predictions)
    if labels_exist:
        _y = np.squeeze(data.y.cpu().detach().numpy())
        unique_labels = torch.unique(torch.tensor(_y))

    _out = np.squeeze(out.max(dim=1)[1].cpu().detach().numpy())


    if labels_exist:
        # Create the mapping
        label_mapping = {}
        for original, normalised in zip(unique_labels, np.unique(_out)):
            label_mapping[original.item()] = normalised.item()

        # Having received y_tensor, use label_mapping
        temporary_list = []
        for y in _y:
            temporary_list.append(label_mapping[y])
        _y = np.array(temporary_list)

        # Record intersections
        intersections = []
        for label, prediction in zip(_y, _out):
            if label == prediction:
                intersections.append(0)
            else:
                intersections.append(10)
        intersections = np.array(intersections)

    mesh = reader.GetOutput()

    # Add data set and write VTK file
    meshNew = dsa.WrapDataObject(mesh)
    meshNew.PointData.append(_out, "Predicted Labels")
    if labels_exist:
        meshNew.PointData.append(intersections, "Intersection between predicted and true labels")

    file_path = os.path.join(folder_path_to_write, tmp_file_name)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(meshNew.VTKObject)
    writer.Write()

    return file_path


if __name__ == '__main__':
    segment("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/GUI/deepl_brain_surfaces/GUI/media/original/data/vtps/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp", './')

