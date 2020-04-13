import os
import os.path as osp
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
# Metrics
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.utils.metric import mean_iou as calculate_mean_iou
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from src.data_loader import OurDataset
from src.utils import get_id, save_to_log, get_comment, get_data_path, data, get_grid_search_local_features
from src.plot_confusion_matrix import plot_confusion_matrix



def add_i_and_u(i, u, i_total, u_total, batch_idx):

    if batch_idx == 0:
        i_total = i
        u_total = u
    else:
        i_total += i
        u_total += u

    return i_total, u_total


def get_mean_iou_per_class(i_total, u_total):

    i_total = i_total.type(torch.FloatTensor)
    u_total = u_total.type(torch.FloatTensor)

    iou_per_class = i_total / u_total
    mean_iou_per_class = torch.sum(iou_per_class, dim=0) / iou_per_class.shape[0]

    return mean_iou_per_class




def get_ious(predictions, labels, batch, batch_idx, num_labels=18):


    # Mean Jaccard index = index averaged over all classes (HENCE, this shows the IoU of a batch) (8 numbers)
    mean_jaccard_indeces = calculate_mean_iou(out.max(dim=1)[1], data.y, 18, batch=data.batch)

    # Mean IoU over classes and batches (1 number)
    mean_ious.append(torch.sum(mean_jaccard_indeces) / len(mean_jaccard_indeces))

    # Mean Jaccard indeces PER LABEL (18 numbers)
    i, u = i_and_u(out.max(dim=1)[1], data.y, 18, batch=data.batch)

    i = i.type(torch.FloatTensor)
    u = u.type(torch.FloatTensor)

    iou_per_class = i / u
    mean_jaccard_index_per_class = torch.sum(iou_per_class, dim=0) / iou_per_class.shape[0]


    return mean_iou, mean_iou_per_label





def mean_iou():
    pass




