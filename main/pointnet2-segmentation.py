import os
import os.path as osp
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
# from deepl_brain_surfaces.shapenet_fake import ShapeNet
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
# Metrics
from torch_geometric.utils import intersection_and_union as i_and_u
from torch_geometric.utils.metric import mean_iou

from src.data_loader import OurDataset
from src.utils import get_id, save_to_log, get_comment, get_data_path, data, get_grid_search_local_features
from src.plot_confusion_matrix import plot_confusion_matrix

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

        # Mean Jaccard index = index averaged over all classes (HENCE, this shows the IoU of a batch)
        mean_jaccard_indeces = mean_iou(out.max(dim=1)[1], data.y, 18, batch=data.batch)

        # Mean Jaccard indeces PER LABEL
        i, u = i_and_u(out.max(dim=1)[1], data.y, 18, batch=data.batch)
        i = i.type(torch.FloatTensor)
        u = u.type(torch.FloatTensor)
        iou_per_class = i/u
        mean_jaccard_index_per_class = torch.sum(iou_per_class, dim=0) / iou_per_class.shape[0]


        if (idx + 1) % 20 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}, Mean IoU: {}'.format(
                idx + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes, np.mean(mean_jaccard_indeces.tolist())))

            # Write to tensorboard: LOSS and IoU per class
            writer.add_scalar('Loss/train', total_loss / 10, epoch)
            writer.add_scalar('Mean IoU/train', torch.sum(mean_jaccard_indeces)/len(mean_jaccard_indeces), epoch)
            writer.add_scalar('Accuracy/train', correct_nodes/total_nodes, epoch)
            for label, iou in enumerate(mean_jaccard_index_per_class):
                writer.add_scalar('IoU{}/train'.format(label), iou, epoch)
                # print('\t\tLabel {}: {}'.format(label, iou))
            # print('\n')
            total_loss = correct_nodes = total_nodes = 0


def test(loader, experiment_description, epoch=None, test=False, id=None, experiment_name=''):

    # 1. Use this as the identifier for testing or validating
    mode = '_validation'
    epoch = str(epoch)
    if test:
        mode = '_test'
        epoch = ''

    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []

    for batch_idx, data in enumerate(loader):

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
        if not os.path.exists(f'experiment_data/{id}-{experiment_name}/'):
            os.makedirs(f'experiment_data/{id}-{experiment_name}/')

        # 4. Save the segmented brain in ./[...comment...]/data_valiation3.pkl (3 is for epoch)
        # for brain_idx, brain in data:
        with open(f'experiment_data/{id}-{experiment_name}/data{mode+epoch}.pkl', 'wb') as file:
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
        writer.add_figure(f'Confusion Matrix - ID{id}-{experiment_name}', cm)

    return loss, accuracy, mean_jaccard_index_per_class



if __name__ == '__main__':

    num_workers = 2
    local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    grid_features = get_grid_search_local_features(local_features)

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################

    data_nativeness = 'aligned'
    data_compression = "50"
    data_type = "inflated"

    additional_comment = ''

    experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{additional_comment}'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    for local_feature_combo in grid_features:
        for global_feature in [[]]:#, ['weight']]:

            # 1. Model Parameters
            lr = 0.001
            batch_size = 8
            global_features = global_feature
            target_class = 'gender'
            task = 'segmentation'
            REPROCESS = True


            # 2. Get the data splits indices
            with open('src/names.pk', 'rb') as f:
                indices = pickle.load(f)


            # 4. Get experiment description
            comment = get_comment(data_nativeness, data_compression, data_type,
                                  lr, batch_size, local_feature_combo, global_features, target_class)


            print(comment)


            # 5. Perform data processing
            data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere='left')

            train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader = data(data_folder,
                                                                                                          files_ending,
                                                                                                          data_type,
                                                                                                          target_class,
                                                                                                          task,
                                                                                                          REPROCESS,
                                                                                                          local_features,
                                                                                                          global_features,
                                                                                                          indices,
                                                                                                          batch_size,
                                                                                                          num_workers=2)


            # 6. Getting the number of features to adapt the architecture
            num_local_features = train_dataset[0].x.size(1)
            # numb_global_features = train_dataset[0].y.size(1) - 1
            num_classes = train_dataset.num_labels

            if not torch.cuda.is_available():
                print('YOU ARE RUNNING ON A CPU!!!!')

            # 7. Create the model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(18, num_local_features, num_global_features=None).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            # 9. Save to log_record.txt
            log_descr = get_comment(data_nativeness, data_compression, data_type,
                                    lr, batch_size, local_feature_combo, global_features, target_class,
                                    log_descr=True)

            save_to_log(log_descr, prefix=experiment_name)
            id = get_id(prefix=experiment_name)

            writer = SummaryWriter(f'runs/{experiment_name}ID' + id)
            writer.add_text(f'{experiment_name} ID #{id}', comment)


            # 10. TRAINING
            for epoch in range(1, 50):

                # 1. Start recording time
                start = time.time()

                # 2. Make a training step
                train(epoch)

                # 3. Validate the performance after each epoch
                loss, acc, iou = test(val_loader, comment+'val'+str(epoch), epoch=epoch, id=id, experiment_name=experiment_name)
                print('Epoch: {:02d}, Val Loss/nll: {}, Val Acc: {:.4f}, Validation IoU (per class):'.format(epoch, loss, acc))

                # 4. Record valiation metrics in Tensorboard
                writer.add_scalar('Loss/val_nll', loss, epoch)
                writer.add_scalar('Accuracy/val', acc, epoch)
                for label, value in enumerate(iou):
                    writer.add_scalar('IoU{}/validation'.format(label), value, epoch)
                    print('\t\tLabel {}: {}'.format(label, value))

                # 5. Stop recording time
                end = time.time()
                print('Time: ' + str(end - start))
                writer.add_scalar('Time/epoch', end-start, epoch)
                print('='*60)

            # 6. Test the performance after training
            loss, acc, iou = test(test_loader, comment, test=True, id=id)

            # 7. Record test metrics in Tensorboard
            writer.add_scalar('Loss/test', loss)
            writer.add_scalar('Accuracy/test', acc)
            for label, value in enumerate(iou):
                writer.add_scalar('IoU{}/test'.format(label), value)
                print('\t\tLabel {}: {}'.format(label, value))

            # 8. Save the model with its unique id
            torch.save(model.state_dict(), f'/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/deepl_brain_surfaces/{id}-{experiment_name}/' + 'model' + '_id' + str(id) + '.pt')



        

























