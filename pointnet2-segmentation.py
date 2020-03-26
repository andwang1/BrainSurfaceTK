import os.path as osp

import torch
import torch.nn.functional as F
# from deepl_brain_surfaces.shapenet_fake import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_interpolate

# Metrics
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.utils.metric import accuracy, precision, recall, f1_score, mean_iou
from sklearn.metrics import confusion_matrix

from data_loader import OurDataset
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from pyvista_examples import plot
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import time
from plot_confusion_matrix import plot_confusion_matrix
from log_saver import get_id, save_to_log
import numpy as np


# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17])


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


# Original network
# class Net(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.sa1_module = SAModule(0.2, 0.2, MLP([3, 64, 64, 128]))
#         self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
#         self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
#
#         self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
#         self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
#         self.fp1_module = FPModule(3, MLP([128, 128, 128, 128]))
#
#         self.lin1 = torch.nn.Linear(128, 128)
#         self.lin2 = torch.nn.Linear(128, 64)
#         self.lin3 = torch.nn.Linear(64, num_classes)
#
#     def forward(self, data):
#         sa0_out = (data.x, data.pos, data.batch)
#         sa1_out = self.sa1_module(*sa0_out)
#         sa2_out = self.sa2_module(*sa1_out)
#         sa3_out = self.sa3_module(*sa2_out)
#
#         fp3_out = self.fp3_module(*sa3_out, *sa2_out)
#         fp2_out = self.fp2_module(*fp3_out, *sa1_out)
#         x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
#
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin3(x)
#         return F.log_softmax(x, dim=-1)


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
#         self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 6, 64, 64, 128]))
#         self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
#         self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
#
#         self.lin1 = Lin(1024, 512)
#         self.lin2 = Lin(512, 256)
#         self.lin3 = Lin(256, 1)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK
#
#     def forward(self, data):
#         sa0_out = (data.x, data.pos, data.batch)
#         sa1_out = self.sa1_module(*sa0_out)
#         sa2_out = self.sa2_module(*sa1_out)
#         sa3_out = self.sa3_module(*sa2_out)
#         x, pos, batch = sa3_out
#
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin3(x)
#         # x FOR REGRESSION, F.log_softmax(x, dim=-1) FOR CLASSIFICATION.
#         return x.view(-1) #F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        # Mean Jaccard index = index averaged over all classes
        mean_jaccard_indeces = mean_iou(out.max(dim=1)[1], data.y, 18, batch=data.batch)

        # Mean Jaccard indeces PER LABEL
        i, u = i_and_u(out.max(dim=1)[1], data.y, 18, batch=data.batch)
        i = i.type(torch.FloatTensor)
        u = u.type(torch.FloatTensor)
        iou_per_class = i/u
        mean_jaccard_index_per_class = torch.sum(iou_per_class, dim=0) / iou_per_class.shape[0]


        if (idx + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}, Mean Jaccard Index (per batch): {}'.format(
                idx + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes, mean_jaccard_indeces))

            # Write to tensorboard: LOSS and IoU per class
            writer.add_scalar('Loss/train', total_loss / 10, epoch)
            writer.add_scalar('Mean IoU/train', torch.sum(mean_jaccard_indeces)/len(mean_jaccard_indeces), epoch)
            for label, iou in enumerate(mean_jaccard_index_per_class):
                writer.add_scalar('IoU{}/train'.format(label), iou, epoch)

            total_loss = correct_nodes = total_nodes = 0


def test(loader, experiment_description, epoch=None, test=False, id=None):

    # 1. Use this as the identifier for testing or validating
    mode = '_validation'
    epoch = str(epoch)
    if test:
        mode = '_test'
        epoch = ''

    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []

    for data in loader:

        # 1. Get predictions and loss
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        loss = F.nll_loss(out, data.y)

        # 2. Get d (positions), _y (actual labels), _out (predictions)
        d = data.pos.cpu().detach().numpy()
        _y = data.y.cpu().detach().numpy()
        _out = out.max(dim=1)[1].cpu().detach().numpy()
        # plot(d, _y, _out)

        # 3. Create directory where to place the data
        if not os.path.exists('./{}/'.format(id)):
            os.makedirs('./{}/'.format(id))

        # 4. Save the segmented brain in ./[...comment...]/data_valiation3.pkl (3 is for epoch)
        with open('./{}/data{}.pkl'.format(id, mode+epoch), 'wb') as file:
            pickle.dump((d, _y, _out), file, protocol=pickle.HIGHEST_PROTOCOL)

        # 5. Get accuracy
        correct_nodes += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes
        accuracy = correct_nodes / total_nodes

        # 6. Get IoU metric per class
        # Mean Jaccard indeces PER LABEL
        i, u = i_and_u(out.max(dim=1)[1], data.y, 18, batch=data.batch)
        i = i.type(torch.FloatTensor)
        u = u.type(torch.FloatTensor)
        iou_per_class = i/u
        mean_jaccard_index_per_class = torch.sum(iou_per_class, dim=0) / iou_per_class.shape[0]

        # 7. Get confusion matrix
        cm = plot_confusion_matrix(data.y, pred, labels=all_labels)
        writer.add_figure('Confusion Matrix', cm)

        # intersections.append(i.to(torch.device('cpu')))
        # unions.append(u.to(torch.device('cpu')))

        # categories.append(data.category.to(torch.device('cpu')))

    # category = torch.cat(categories, dim=0)


    # intersection = torch.cat(intersections, dim=0)
    # union = torch.cat(unions, dim=0)
    #
    # ious = []
    # for j in range(len(loader.dataset)):
    #     i = intersection[j, loader.dataset.y_mask[category[j]]]
    #     u = union[j, loader.dataset.y_mask[category[j]]]
    #     iou = i.to(torch.float) / u.to(torch.float)
    #     iou[torch.isnan(iou)] = 1
    #     ious[category[j]].append(iou.mean().item())
    #
    # for cat in range(len(loader.dataset.categories)):
    #     ious[cat] = torch.tensor(ious[cat]).mean().item()

    return loss, accuracy, mean_jaccard_index_per_class


if __name__ == '__main__':

    # Model Parameters
    lr = 0.001
    batch_size = 8
    num_workers = 2
    local_features = ['corr_thickness', 'myelin_map']#, 'curvature']#, 'sulc']
    global_features = ['weight']
    # global_features = None
    target_class = 'gender'
    task = 'segmentation'
    id = get_id()
    # number_of_points = 12000

    test_size = 0.09
    val_size = 0.1
    reprocess = False

    data = "reduced_50"
    type_data = "inflated"

    comment = "LR_" + str(lr) \
              + "_BATCH_" + str(batch_size) \
              + "_NUM_WORKERS_" + str(num_workers) \
              + "_local_features_" + str(local_features) \
              + "_global_features_" + str(global_features) \
              + data + "_" + type_data

    # Tensorboard writer.
    writer = SummaryWriter(comment='ID' + get_id() + '_' + comment)

    print(comment)

    # 0. Save to log_record.txt
    log_descr = "LR=" + str(lr) + '\t\t'\
              + "Batch=" + str(batch_size) + '\t\t'\
              + "Num Workers=" + str(num_workers) + '\t\t'\
              + "Local features:" + str(local_features) + '\t\t'\
              + "Global features:" + str(global_features) + '\t\t'\
              + "Data used: " + data + '_' + type_data + '\t\t'\
              + "Split class: " + target_class

    save_to_log(log_descr)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/' + target_class + '/Reduced50/inflated')

    # Transformations
    transform = T.Compose([
        # T.RandomTranslate(0.1),
        # T.RandomFlip(0, p=0.3),
        # T.RandomFlip(1, p=0.1),
        # T.RandomFlip(2, p=0.3),
        T.RandomRotate(30, axis=0),
        # T.RandomRotate(15, axis=1),
        # T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()

    # DATASETS
    train_dataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform,
                                                 target_class=target_class, task=task, reprocess=reprocess,
                                                 local_features=local_features, global_feature=global_features,
                                                 test_size=test_size, val_size=val_size, val=False)

    test_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
                                                 target_class=target_class, task=task, reprocess=reprocess,
                                                 local_features=local_features, global_feature=global_features,
                                                 test_size=test_size, val_size=val_size, val=False)

    validation_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
                                                 target_class=target_class, task=task, reprocess=reprocess,
                                                 local_features=local_features, global_feature=global_features,
                                                 test_size=test_size, val_size=val_size, val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Getting the number of features to adapt the architecture
    num_local_features = train_dataset[0].x.size(1)
    # numb_global_features = train_dataset[0].y.size(1) - 1
    num_classes = train_dataset.num_labels

    if not torch.cuda.is_available():
        print('YOU ARE RUNNING ON A CPU!!!!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(18, num_local_features, num_global_features=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, 50):

        # 1. Start recording time
        start = time.time()

        # 2. Make a training step
        train(epoch)

        # 3. Validate the performance after each epoch
        loss, acc, iou = test(val_loader, comment+'val'+str(epoch), epoch=epoch, id=id)
        print('Epoch: {:02d}, Val Loss/nll: {}, Val Acc: {:.4f}'.format(epoch, loss, acc))

        # 4. Record valiation metrics in Tensorboard
        writer.add_scalar('Loss/val_nll', loss, epoch)
        for label, value in enumerate(iou):
            writer.add_scalar('IoU{}/validation'.format(label), value, epoch)

        # 5. Stop recording time
        end = time.time()
        print('Time: ' + str(end - start))
        writer.add_scalar('Time/epoch', end-start, epoch)

    # 6. Test the performance after training
    loss, acc, iou = test(test_loader, comment, test=True, id=id)

    # 7. Record test metrics in Tensorboard
    writer.add_scalar('Loss/test', loss)
    writer.add_scalar('Accuracy/test', acc)
    for label, value in enumerate(iou):
        writer.add_scalar('IoU{}/test'.format(label), value)

    # 8. Save the model with its unique id
    torch.save(model.state_dict(), './{}/'.format(get_id()) + 'model' + '_id' + get_id() + '.pt')





























