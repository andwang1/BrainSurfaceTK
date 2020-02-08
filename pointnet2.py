import os.path as osp

import torch
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import time
from data_loader import OurDataset
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self):
        super(Net, self).__init__()
        # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 6, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # That +1 is for weight at birth
        self.lin1 = Lin(1024 + 1, 512)
        # self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 1)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Adding weight at birth. #  TODO: Do this properly for all the features. (Might leave it manual?)
        x = torch.cat((x, data.y[1].expand((x.size(0), 1))), 1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        # x.view(-1) FOR REGRESSION, F.log_softmax(x, dim=-1) FOR CLASSIFICATION.
        return x.view(-1) #F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()
    loss_train = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # USE F.nll_loss FOR CLASSIFICATION, F.mse_loss FOR REGRESSION.
        # loss = F.nll_loss(model(data), data.y)
        pred = model(data)
        loss = F.mse_loss(pred, data.y[0].expand(pred.size()))
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

        correct += pred.eq(data.y[0]).sum().item()
    return correct / len(loader.dataset)


def test_regression(loader):
    model.eval()

    mse = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            # print(torch.sum((pred - data.y) ** 2) / len(pred))
            print(pred.t(), data.y[0].t())
            loss_test = F.mse_loss(pred, data.y[0].expand(pred.size()))
        # mse += torch.sum((pred - data.y) ** 2) / len(pred)
        mse += loss_test.item()
    return mse / len(loader)


if __name__ == '__main__':

    # Model Parameters
    lr = 0.001
    batch_size = 1
    num_workers = 2
    add_birth_weight = True
    # Additional comments
    comment = ""

    # Tensorboard writer.
    writer = SummaryWriter(comment="LR_"+str(lr)+"_BATCH_"+str(batch_size)
                                   +"_NUM_WORKERS_"+str(num_workers)+"_ADD_BIRTH_WEIGHT_"
                                   +str(add_birth_weight)+"_Comment_"+comment)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data')

    # DEFINE TRANSFORMS HERE.
    # 32492
    transform = T.Compose([
        T.FixedPoints(15000)
        # T.SamplePoints(1024)  # THIS ONE DOESN'T KEEP FEATURES(x)
    ])

    # TRANSFORMS DONE BEFORE SAVING THE DATA IF THE DATA IS NOT YET PROCESSED.
    pre_transform = T.NormalizeScale()

    # TODO: DEFINE TEST/TRAIN SPLIT (WHEN MORE DATA IS AVAILABLE). NOW TESTING == TRAINING
    train_dataset = OurDataset(path, label_class='scan_age', train=True, classification=False,
                                 transform=transform, pre_transform=pre_transform, add_birth_weight=add_birth_weight)
    test_dataset = OurDataset(path, label_class='scan_age', train=False, classification=False,
                                 transform=transform, pre_transform=pre_transform, add_birth_weight=add_birth_weight)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if not torch.cuda.is_available():
        print('YOU ARE RUNNING ON A CPU!!!!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # MAIN TRAINING LOOP
    for epoch in range(1, 11):
        start = time.time()
        train(epoch)
        test_acc = test_regression(test_loader)

        writer.add_scalar('Loss/test', test_acc, epoch)

        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
        end = time.time()
        print('Time: ' + str(end - start))
