
model_path = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/deepl_brain_surfaces/experiment_data/new/aligned_inflated_50_-7/best_acc_model.pt'

import os
import os.path as osp
import pickle
import time
import os.path as osp
import torch
from torch_geometric.data import Data
# import pyvista as pv
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
# from pyvista import read
import pandas as pd
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate



# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17])
num_points_dict = {'original': 32492, '50': 16247}

recording = True

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


# My network
class Net(torch.nn.Module):
    def __init__(self, num_classes, num_local_features, num_global_features=None):
        '''
        :param num_classes: Number of segmentation classes
        :param num_local_features: Feature per node
        :param num_global_features: NOT USED
        '''
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + num_local_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + num_local_features, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)




def get_features(list_features, reader):
    '''Returns tensor of features to add in every point.
    :param list_features: list of features to add. Mapping is in self.feature_arrays
    :param mesh: pyvista mesh from which to get the arrays.
    :returns: tensor of features or None if list is empty.'''

    # Very ugly workaround about some classes not being in some data.
    list_of_drawem_labels = [0, 5, 7, 9, 11, 13, 15, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}

    if list_features:
        drawem_list = []

        # features = [mesh.get_array(feature_arrays[key]) for key in feature_arrays if key != 'drawem']
        features = [vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(feature_arrays[key])) for key in feature_arrays if key != 'drawem']

        return torch.tensor(features + drawem_list).t()
    else:
        return None




def segment(brain_path=None):

    torch.manual_seed(0)
    if osp.exists(brain_path):

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(brain_path)
        reader.Update()

        points = torch.tensor(np.array(reader.GetOutput().GetPoints().GetData()))

        local_features = ['corr_thickness', 'curvature', 'sulc']
        x = get_features(local_features, reader)

        data = Data(batch=torch.zeros_like(x[:, 0]).long(), x=x, pos=points)
        # data = Data(x=x, pos=points)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        numb_local_features = x.size(1)
        numb_global_features = 0

        model = Net(18, numb_local_features).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()


        # 1. Get predictions and loss
        data = data.to(device)
        out = model(data)

        # 2. Get d (positions), _y (actual labels), _out (predictions)
        _out = out.max(dim=1)[1].cpu().detach().numpy()

        mesh = reader.GetOutput()

        # Add data set and write VTK file
        meshNew = dsa.WrapDataObject(mesh)
        meshNew.PointData.append(_out, "New Labels")

        path_to_write = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/GUI/deepl_brain_surfaces/GUI/backend/'
        file_name = 'segmented_brain.vtp'
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path_to_write + file_name)
        writer.SetInputData(meshNew.VTKObject)
        writer.Write()

        return path_to_write + file_name


if __name__ == '__main__':

    segment("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/GUI/deepl_brain_surfaces/GUI/media/original/data/vtps/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp")

