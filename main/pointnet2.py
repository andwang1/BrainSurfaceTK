import os.path as osp
import time

import torch
import torch.nn.functional as F
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
        self.lin4 = Lin(128, 1)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1a_out = self.sa1a_module(*sa1_out)
        sa2_out = self.sa2_module(*sa1a_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Concatenates global features to the inputs.
        if self.num_global_features > 0:
            x = torch.cat((x, data.y[:, 1:self.num_global_features+1].view(-1, self.num_global_features)), 1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        # x.view(-1) FOR REGRESSION, F.log_softmax(x, dim=-1) FOR CLASSIFICATION.
        return x.view(-1) #F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()
    loss_train = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # USE F.nll_loss FOR CLASSIFICATION, F.mse_loss FOR REGRESSION.
        pred = model(data)
        loss = F.mse_loss(pred, data.y[:, 0])
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    writer.add_scalar('Loss/train', loss_train / len(train_loader), epoch)

def test_classification(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
            print(pred.t(), data.y[:, 0])
        correct += pred.eq(data.y[:, 0].long()).sum().item()
    return correct / len(loader.dataset)


def test_regression(loader):
    model.eval()

    mse = 0
    l1 = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            print(pred.t(), data.y[:, 0])
            loss_test_mse = F.mse_loss(pred, data.y[:, 0])
            loss_test_l1 = F.l1_loss(pred, data.y[:, 0])
        mse += loss_test_mse.item()
        l1 += loss_test_l1.item()
    return mse / len(loader), l1 / len(loader)


if __name__ == '__main__':

    # Model Parameters
    lr = 0.001
    batch_size = 8
    num_workers = 2

    local_features = ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']
    global_features = ['weight']
    target_class = 'scan_age'
    task = 'regression'
    number_of_points = 12000

    test_size = 0.09
    val_size = 0.1
    reprocess = True

    data = "reduced_50"
    type_data = "inflated"

    comment = "LR_" + str(lr) \
            + "_BATCH_" + str(batch_size) \
            + "_NUM_WORKERS_" + str(num_workers)\
            + "_local_features_" + str(local_features)\
            + "_glogal_features_" + str(global_features) \
            + "number_of_points" + str(number_of_points)\
            + data + "_" + type_data

    # Tensorboard writer.
    writer = SummaryWriter(comment=comment)

    print(comment)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/'+target_class+'/Reduced50/inflated')

    # DEFINE TRANSFORMS HERE.
    transform = T.Compose([
        T.FixedPoints(number_of_points)
    ])

    # TRANSFORMS DONE BEFORE SAVING THE DATA IF THE DATA IS NOT YET PROCESSED.
    pre_transform = T.NormalizeScale()

    # Creating datasets and dataloaders for train/test/val.
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
    numb_local_features = train_dataset[0].x.size(1)
    numb_global_features = train_dataset[0].y.size(1) - 1

    if not torch.cuda.is_available():
        print('YOU ARE RUNNING ON A CPU!!!!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(numb_local_features, numb_global_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # MAIN TRAINING LOOP
    for epoch in range(1, 51):
        start = time.time()
        train(epoch)
        test_mse, test_l1 = test_regression(val_loader)

        writer.add_scalar('Loss/val_mse', test_mse, epoch)
        writer.add_scalar('Loss/val_l1', test_l1, epoch)

        print('Epoch: {:03d}, Test loss l1: {:.4f}'.format(epoch, test_l1))
        end = time.time()
        print('Time: ' + str(end - start))
        writer.add_scalar('Time/epoch', end - start, epoch)

    writer.add_scalar('Final Test Loss (L1)', test_regression(test_loader))

    # save the model
    torch.save(model.state_dict(), 'model' + comment + '.pt')
