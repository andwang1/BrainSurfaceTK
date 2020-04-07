import os.path as osp
import os
import time
import pickle
import csv

import datetime as datetime
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from src.data_loader import OurDataset


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)  # TODO: FIGURE OUT THIS WITH RESPECT TO NUMBER OF POINTS
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


class Net(torch.nn.Module):
    def __init__(self, num_local_features, num_global_features):
        super(Net, self).__init__()

        self.num_global_features = num_global_features

        # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_local_features, 64, 64, 96]))
        self.sa1a_module = SAModule(0.5, 0.2, MLP([96 + 3, 96, 96, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024 + num_global_features, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)
        self.lin4 = Lin(128, 2)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1a_out = self.sa1a_module(*sa1_out)
        sa2_out = self.sa2_module(*sa1a_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Concatenates global features to the inputs.
        if self.num_global_features > 0:
            x = torch.cat((x, data.y[:, 1:self.num_global_features + 1].view(-1, self.num_global_features)), 1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        # x.view(-1) FOR REGRESSION, F.log_softmax(x, dim=-1) FOR CLASSIFICATION.
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()
    loss_train = 0.0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # USE F.nll_loss FOR CLASSIFICATION, F.mse_loss FOR REGRESSION.
        pred = model(data)
        perd_label = model(data).max(1)[1]
        loss = F.nll_loss(pred, data.y[:, 0])
        loss.backward()
        optimizer.step()
        correct += perd_label.eq(data.y[:, 0].long()).sum().item()
        loss_train += loss.item()
        #print(str(perd_label), str(data.y[:, 0]))
    acc = correct / len(train_loader.dataset)
    writer.add_scalar('Acc/train', acc, epoch)
    writer.add_scalar('Loss/train', loss_train / len(train_loader), epoch)
    print('Train acc: ' + str(acc))


def test_classification(loader, indices, results_folder, val=True, epoch=0):
    model.eval()
    with open(results_folder + '/results.csv', 'a', newline='') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if val:
            print('Validation'.center(60, '-'))
            result_writer.writerow(['Val accuracy Epoch - ' + str(epoch)])
        else:
            print('Test'.center(60, '-'))
            result_writer.writerow(['Test accuracy'])

        correct = 0
        for idx, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
                print(str(pred.t().item()).center(20, ' '), str(data.y[:, 0].item()).center(20, ' '), indices[idx])
                result_writer.writerow([indices[idx][:11], indices[idx][12:],
                                        str(pred.t().item()), str(data.y[:, 0].item())])
            correct += pred.eq(data.y[:, 0].long()).sum().item()

        acc = correct / len(loader.dataset)

        if val:
            result_writer.writerow(['Epoch average error:', str(acc)])
        else:
            result_writer.writerow(['Test average error:', str(acc)])

    return acc


if __name__ == '__main__':

    # Model Parameters
    lr = 0.001
    batch_size = 8
    num_workers = 4
    # ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc'] + ['weight']
    local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    # local_features = []
    global_features = []
    # target_class = 'scan_age'
    target_class = 'gender'
    task = 'classification'
    number_of_points = 3251  # 3251# 12000  # 16247

    # For quick tests
    # indices = {'Train': ['CC00050XX01_7201', 'CC00050XX01_7201'],
    #            'Test': ['CC00050XX01_7201', 'CC00050XX01_7201'],
    #            'Val': ['CC00050XX01_7201', 'CC00050XX01_7201']}

    reprocess = False

    # inflated  midthickness  pial  sphere  veryinflated  white

    # data = "reduced_50"
    # data_ending = "reduce50.vtk"
    # type_data = "inflated"

    data = "reduced_90"
    data_ending = "reduce90.vtk"
    type_data = "pial"
    native = "surface_fsavg32k"  # "surface_native" #surface_fsavg32k
    #
    # data = "reduced_50"
    # data_ending = "reduce50.vtk"
    # type_data = "pial"
    #
    # data = "reduced_50"
    # data_ending = "reduce50.vtk"
    # type_data = "white"

    # folder in data/stored for data.
    stored = target_class + '/' + type_data + '/' + data + '/' + str(local_features + global_features) + '/' + native

    data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/" + native + "/" + data \
                  + "/vtk/" + type_data

    #    data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/" + native + "/" + data \
    #                  + "/" + type_data + "/vtk"
    print(data_folder)
    # sub-CC00466AN13_ses-138400_right_pial_reduce90.vtk
    files_ending = "_hemi-L_" + type_data + "_" + data_ending
    #    files_ending = "_left_" + type_data + "_" + data_ending

    # From quick local test
    #data_folder = "/home/vital/Group Project/deepl_brain_surfaces/random"

    with open('src/names.pk', 'rb') as f:
        indices = pickle.load(f)

    # TESTING PURPOSES
    # indices['Train'] = indices['Train'][:2]

    comment = 'Gender' + str(datetime.datetime.now()) \
              + "__LR__" + str(lr) \
              + "__BATCH_" + str(batch_size) \
              + "__local_features__" + str(local_features) \
              + "__glogal_features__" + str(global_features) \
              + "__number_of_points__" + str(number_of_points) \
              + "__" + data + "__" + type_data + '__no_rotate'

    results_folder = 'runs/' + task + '/' + comment + '/results'
    model_dir = 'runs/' + task + '/' + comment + '/models'

    if not osp.exists(results_folder):
        os.makedirs(results_folder)

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    with open(results_folder + '/configuration.txt', 'w', newline='') as config_file:
        config_file.write('Learning rate - ' + str(lr) + '\n')
        config_file.write('Batch size - ' + str(batch_size) + '\n')
        config_file.write('Local features - ' + str(local_features) + '\n')
        config_file.write('Global feature - ' + str(global_features) + '\n')
        config_file.write('Number of points - ' + str(number_of_points) + '\n')
        config_file.write('Data res - ' + data + '\n')
        config_file.write('Data type - ' + type_data + '\n')
        config_file.write('Additional comments - With no rotate transforms' + '\n')

    with open(results_folder + '/results.csv', 'w', newline='') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label'])

    # Tensorboard writer.
    writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/' + stored)

    # DEFINE TRANSFORMS HERE.
    transform = T.Compose([
        T.FixedPoints(number_of_points),
        # T.RandomRotate(360, axis=0),
        # T.RandomRotate(360, axis=1),
        # T.RandomRotate(360, axis=2)
    ])

    # TRANSFORMS DONE BEFORE SAVING THE DATA IF THE DATA IS NOT YET PROCESSED.
    pre_transform = T.NormalizeScale()

    # Creating datasets and dataloaders for train/test/val.
    train_dataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform, val=False,
                               target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
                               local_features=local_features, global_feature=global_features,
                               indices=indices['Train'], data_folder=data_folder)

    test_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform, val=False,
                              target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
                              local_features=local_features, global_feature=global_features,
                              indices=indices['Test'], data_folder=data_folder)

    val_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform, val=True,
                             target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
                             local_features=local_features, global_feature=global_features,
                             indices=indices['Val'], data_folder=data_folder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Getting the number of features to adapt the architecture
    if len(local_features) > 0:
        numb_local_features = train_dataset[0].x.size(1)
    else:
        numb_local_features = 0
    numb_global_features = len(global_features)

    if not torch.cuda.is_available():
        print('YOU ARE RUNNING ON A CPU!!!!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(numb_local_features, numb_global_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.985)

    best_val_acc = 0.0

    # MAIN TRAINING LOOP
    for epoch in range(1, 201):
        start = time.time()
        train(epoch)
        val_acc = test_classification(val_loader, indices['Val'], results_folder, epoch=epoch)

        scheduler.step()

        writer.add_scalar('Acc/val', val_acc, epoch)

        print('Epoch: {:03d}, Test acc: {:.4f}'.format(epoch, val_acc))
        end = time.time()
        print('Time: ' + str(end - start))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir + '/model_best.pt')
            print('Saving Model'.center(60, '-'))
        writer.add_scalar('Time/epoch', end - start, epoch)

    test_classification(test_loader, indices['Test'], results_folder)

    # save the last model
    torch.save(model.state_dict(), model_dir + '/model_last.pt')

    # Eval best model on test
    model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

    with open(results_folder + '/results.csv', 'a', newline='') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Best model!'])
    test_classification(test_loader, indices['Test'], results_folder)

