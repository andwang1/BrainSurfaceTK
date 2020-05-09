import os
import os.path as osp
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate

# Metrics
from torch_geometric.utils import intersection_and_union as i_and_u
from torch.optim.lr_scheduler import StepLR

from ..src.metrics import add_i_and_u, get_mean_iou_per_class
from ..src.utils import get_id, save_to_log, get_comment, get_data_path, data, get_grid_search_local_features
from ..src.models.pointnet2_segmentation import Net

# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17])
num_points_dict = {'original': 32492, '50': 16247, '90': None}
recording = True
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'


def train(model, train_loader, epoch, device, optimizer, num_labels, writer, recording=False):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    i_total, u_total = None, None
    print_per = 10

    for idx, data in enumerate(train_loader):

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        pred = out.max(dim=1)[1]

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        # Mean Jaccard indeces PER LABEL (18 numbers)
        i, u = i_and_u(out.max(dim=1)[1], data.y, num_labels, batch=data.batch)

        # Add to totals
        i_total, u_total = add_i_and_u(i, u, i_total, u_total, idx)

        if (idx + 1) % print_per == 0:

            mean_iou_per_class = get_mean_iou_per_class(i_total, u_total)
            mean_iou = torch.tensor(np.nanmean(mean_iou_per_class.cpu().detach().numpy()))

            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}, Mean IoU: {}'.format(
                idx + 1, len(train_loader), total_loss / print_per,
                correct_nodes / total_nodes, mean_iou))

            # Write to tensorboard: LOSS and IoU per class
            if recording:
                writer.add_scalar('Loss/train', total_loss / print_per, epoch)
                writer.add_scalar('Mean IoU/train', mean_iou, epoch)
                writer.add_scalar('Accuracy/train', correct_nodes/total_nodes, epoch)

                for label, iou in enumerate(mean_iou_per_class):
                    writer.add_scalar('IoU{}/train'.format(label), iou, epoch)

                # print('\t\tLabel {}: {}'.format(label, iou))
            # print('\n')
            total_loss = correct_nodes = total_nodes = 0
            i_total, u_total = torch.zeros_like(i_total), torch.zeros_like(u_total)


def test(model, loader, experiment_description, device, num_labels, writer, epoch=None, test=False, test_by_acc_OR_iou='acc', id=None, experiment_name='', recording=False):

    # 1. Use this as the identifier for testing or validating
    mode = '_validation'
    _epoch = str(epoch)
    if test:
        mode = '_test'
        _epoch = ''

    model.eval()

    correct_nodes = total_nodes = 0
    i_total, u_total = None, None
    intersections, unions, categories = [], [], []
    all_preds = None
    all_datay = None
    total_loss = []

    start = time.time()


    with torch.no_grad():

        for batch_idx, data in enumerate(loader):

            # 1. Get predictions and loss
            data = data.to(device)
            out = model(data)

            pred = out.max(dim=1)[1]

            loss = F.nll_loss(out, data.y)
            total_loss.append(loss)


            # 2. Get d (positions), _y (actual labels), _out (predictions)
            d = data.pos.cpu().detach().numpy()
            _y = data.y.cpu().detach().numpy()
            _out = out.max(dim=1)[1].cpu().detach().numpy()


            # Mean Jaccard indeces PER LABEL (18 numbers)
            i, u = i_and_u(out.max(dim=1)[1], data.y, num_labels, batch=data.batch)
            i_total, u_total = add_i_and_u(i, u, i_total, u_total, batch_idx)


            if batch_idx == 0:
                all_preds = pred
                all_datay = data.y

            else:
                all_preds = torch.cat((all_preds, pred))
                all_datay = torch.cat((all_datay, data.y))


            if recording:
                # 3. Create directory where to place the data
                if not os.path.exists(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/'):
                    print(f'Created folder: experiment_data/new/{experiment_name}-{id}/')
                    os.makedirs(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/')

                # 4. Save the segmented brain in ./[...comment...]/data_valiation3.pkl (3 is for epoch)
                # for brain_idx, brain in data:
                if test:
                    with open(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/data{mode}_by_{test_by_acc_OR_iou}.pkl', 'wb') as file:
                        pickle.dump((d, _y, _out), file, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/data{mode+_epoch}.pkl', 'wb') as file:
                        pickle.dump((d, _y, _out), file, protocol=pickle.HIGHEST_PROTOCOL)


            # 5. Get accuracy
            correct_nodes += pred.eq(data.y).sum().item()
            total_nodes += data.num_nodes

        # Mean IoU over all batches and per class (i.e. array of shape 18 - [0.5, 0.7, 0.85, ... ]
        mean_iou_per_class = get_mean_iou_per_class(i_total, u_total)
        mean_iou = torch.tensor(np.nanmean(mean_iou_per_class.cpu().detach().numpy()))

        accuracy = correct_nodes / total_nodes
        loss = torch.mean(torch.tensor(total_loss))

        if recording:

            if test:
                writer.add_scalar(f'Mean IoU/test_by_{test_by_acc_OR_iou}', mean_iou)
            else:
                writer.add_scalar(f'Mean IoU/validation', mean_iou, epoch)


            writer.add_scalar('Validation Time/epoch', time.time() - start, epoch)

            # 7. Get confusion matrix
            # cm = plot_confusion_matrix(all_datay, all_preds, labels=all_labels)
            # writer.add_figure(f'Confusion Matrix - ID{id}-{experiment_name}', cm)

    return loss, accuracy, mean_iou_per_class, mean_iou


def perform_final_testing(model, writer, test_loader, experiment_name, comment, id, num_labels, device, best_model_acc, best_model_iou, recording=False):

    # 1. Load the best model for testing --- both by accuracy and IoU
    model.load_state_dict(
        torch.load(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/' + 'best_acc_model' + '.pt'))

    # 2. Test the performance after training
    loss_acc, acc_acc, iou_acc, mean_iou_acc = test(model, test_loader, comment, device, num_labels, writer, test=True, test_by_acc_OR_iou='acc', id=id, experiment_name=experiment_name, recording=recording)

    # 3. Record test metrics in Tensorboard
    if recording:
        writer.add_scalar('Loss/test_byACC', loss_acc)
        writer.add_scalar('Accuracy/test_byACC', acc_acc)
        print(f'****************** Loaded best model by Acc. It was saved at epoch {best_model_acc} ******************')
        for label, value in enumerate(iou_acc):
            writer.add_scalar('IoU{}/test_byACC'.format(label), value)
            print('\t\tTest Label (best model by Acc) {}: {}'.format(label, value))



    # 1. Load the best model for testing --- both by accuracy and IoU
    model.load_state_dict(
        torch.load(PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/' + 'best_iou_model' + '.pt'))

    # 2. Test the performance after training
    loss_iou, acc_iou, iou_iou, mean_iou_iou = test(model, test_loader, comment, device, num_labels, writer, test=True, test_by_acc_OR_iou='iou', id=id, experiment_name=experiment_name, recording=recording)

    # 3. Record test metrics in Tensorboard
    if recording:
        writer.add_scalar('Loss/test_byIOU', loss_iou)
        writer.add_scalar('Accuracy/test_byIOU', acc_iou)
        print(f'****************** Loaded best model by IoU. It was saved at epoch {best_model_iou} ******************')
        for label, value in enumerate(iou_iou):
            writer.add_scalar('IoU{}/test_byIOU'.format(label), value)
            print('\t\tTest Label (best model by IoU) {}: {}'.format(label, value))

    return loss_acc, acc_acc, iou_acc, mean_iou_acc, \
           loss_iou, acc_iou, iou_iou, mean_iou_iou





if __name__ == '__main__':

    num_workers = 2
    local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    grid_features = get_grid_search_local_features(local_features)
    ids_to_include = [15]

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = True

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'white'
    hemisphere = 'left'

    additional_comment = ''

    experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{hemisphere}_{additional_comment}'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################


    for id in ids_to_include:
        for global_feature in [[]]:#, ['weight']]:

            # 1. Model Parameters
            lr = 0.001
            batch_size = 4
            local_feature_combo = grid_features[id-2]
            global_features = global_feature
            target_class = 'gender'
            task = 'segmentation'
            REPROCESS = True


            # 2. Get the data splits indices
            with open(PATH_TO_ROOT + 'src/names.pk', 'rb') as f:
                indices = pickle.load(f)


            # 4. Get experiment description
            comment = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                                  lr, batch_size, local_feature_combo, global_features, target_class)

            print('='*50 + '\n' + '='*50)
            print(comment)
            print('='*50 + '\n' + '='*50)

            # 5. Perform data processing
            data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere=hemisphere)

            train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(data_folder,
                                                                                                                      files_ending,
                                                                                                                      data_type,
                                                                                                                      target_class,
                                                                                                                      task,
                                                                                                                      REPROCESS,
                                                                                                                      local_feature_combo,
                                                                                                                      global_features,
                                                                                                                      indices,
                                                                                                                      batch_size,
                                                                                                                      num_workers=2,
                                                                                                                      data_nativeness=data_nativeness,
                                                                                                                      data_compression=data_compression,
                                                                                                                      hemisphere=hemisphere
                                                                                                                      )

            # 6. Getting the number of features to adapt the architecture
            try:
                num_local_features = train_dataset[0].x.size(1)
            except:
                num_local_features = 0
            # numb_global_features = train_dataset[0].y.size(1) - 1

            print(f'Unique labels found: {num_labels}')

            if not torch.cuda.is_available():
                print('YOU ARE RUNNING ON A CPU!!!!')

            # 7. Create the model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(num_labels, num_local_features, num_global_features=None).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            gamma = 0.9875
            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

            id = '0'
            if recording:
                # 9. Save to log_record.txt
                log_descr = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                                        lr, batch_size, local_feature_combo, global_features, target_class,
                                        log_descr=True)

                save_to_log(log_descr, prefix=experiment_name)

                id = str(int(get_id(prefix=experiment_name)) - 1)

                writer = SummaryWriter(PATH_TO_ROOT + f'new_runs/{experiment_name}ID' + id)
                writer.add_text(f'{experiment_name} ID #{id}', comment)

            best_val_acc = 0
            best_val_iou = 0
            best_model_acc = 0
            best_model_iou = 0
            # 10. TRAINING
            for epoch in range(1, 150):

                # 1. Start recording time
                start = time.time()

                # 2. Make a training step
                train(model, train_loader, epoch, device, optimizer, num_labels, writer, recording=recording)

                if recording:
                    writer.add_scalar('Training Time/epoch', time.time() - start, epoch)

                # 3. Validate the performance after each epoch
                loss, acc, iou, mean_iou = test(model, val_loader, comment+'val'+str(epoch), device, num_labels, writer, epoch=epoch, id=id, experiment_name=experiment_name, recording=recording)
                print('Epoch: {:02d}, Val Loss/nll: {}, Val Acc: {:.4f}'.format(epoch, loss, acc))

                scheduler.step()

                # 4. Record valiation metrics in Tensorboard
                if recording:

                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_model_acc = epoch
                        torch.save(model.state_dict(), PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/' + 'best_acc_model' + '.pt')
                    if mean_iou > best_val_iou:
                        best_val_iou = mean_iou
                        best_model_iou = epoch
                        torch.save(model.state_dict(), PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/' + 'best_iou_model' + '.pt')

                    writer.add_scalar('Loss/val_nll', loss, epoch)
                    writer.add_scalar('Accuracy/val', acc, epoch)
                    for label, value in enumerate(iou):
                        writer.add_scalar('IoU{}/validation'.format(label), value, epoch)
                        print('\t\tValidation Label {}: {}'.format(label, value))

                print('='*60)

            # save the last model
            torch.save(model.state_dict(), PATH_TO_ROOT + f'experiment_data/new/{experiment_name}-{id}/' + 'last_model' + '.pt')


            loss_acc, acc_acc, iou_acc, mean_iou_acc, loss_iou, acc_iou, iou_iou, mean_iou_iou = perform_final_testing(model, writer, test_loader,
                                                                                                                       experiment_name, comment, id,
                                                                                                                       num_labels, writer, recording=recording)





        

























