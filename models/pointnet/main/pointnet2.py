import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import os
import time
import pickle
import csv

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.pointnet.src.models.pointnet2_regression_v2 import Net
from models.pointnet.src.utils import get_data_path, data


def train(model, train_loader, epoch, device, optimizer, scheduler, writer):
    model.train()
    loss_train = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y[:, 0])
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    scheduler.step()

    if writer is not None:
        writer.add_scalar('Loss/train_mse', loss_train / len(train_loader), epoch)


def test_regression(model, loader, indices, device, recording, results_folder, val=True, epoch=0):
    model.eval()
    if recording:

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if val:
                print('Validation'.center(60, '-'))
                result_writer.writerow(['Val scores Epoch - ' + str(epoch)])
            else:
                print('Test'.center(60, '-'))
                result_writer.writerow(['Test scores'])

            mse = 0
            l1 = 0
            for idx, data in enumerate(loader):

                data = data.to(device)
                with torch.no_grad():
                    pred = model(data)
                    for i in range(len(pred)):
                        print(str(pred[i].item()).center(20, ' '),
                              str(data.y[:, 0][i].item()).center(20, ' '),
                              indices[idx * len(pred) + i])

                        result_writer.writerow([indices[idx * len(pred) + i][:11], indices[idx * len(pred) + i][12:],
                                                str(pred[i].item()), str(data.y[:, 0][i].item()),
                                                str(abs(pred[i].item() - data.y[:, 0][i].item()))])

                    loss_test_mse = F.mse_loss(pred, data.y[:, 0])
                    loss_test_l1 = F.l1_loss(pred, data.y[:, 0])
                    mse += loss_test_mse.item()
                    l1 += loss_test_l1.item()
            if val:
                result_writer.writerow(['Epoch average error:', str(l1 / len(loader))])
                print(f'Epoch {epoch} average error: {l1 / len(loader)}')
            else:
                result_writer.writerow(['Test average error:', str(l1 / len(loader))])
                print(f'Test average error: {l1 / len(loader)}')
    else:

        if val:
            print('Validation'.center(60, '-'))
        else:
            print('Test'.center(60, '-'))

        mse = 0
        l1 = 0
        for idx, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)

                for i in range(len(pred)):
                    print(str(pred[i].item()).center(20, ' '),
                          str(data.y[:, 0][i].item()).center(20, ' '),
                          indices[idx * len(pred) + i])

                loss_test_mse = F.mse_loss(pred, data.y[:, 0])
                loss_test_l1 = F.l1_loss(pred, data.y[:, 0])
                mse += loss_test_mse.item()
                l1 += loss_test_l1.item()

        if val:
            print(f'Epoch {epoch} average error (L1): {l1 / len(loader)}')
        else:
            print(f'Test average error (L1): {l1 / len(loader)}')

    return mse / len(loader), l1 / len(loader)


if __name__ == '__main__':

    PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'

    num_workers = 2
    local_features = []
    global_features = []

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = False
    REPROCESS = True

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'white'
    hemisphere = 'left'

    comment = 'comment1'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    ################################################
    lr = 0.001
    batch_size = 1
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'scan_age'
    task = 'regression'
    numb_epochs = 200
    number_of_points = 10000

    ################################################
    ########## INDICES FOR DATA SPLIT #############
    with open(PATH_TO_ROOT + 'src/names_06152020_noCrashSubs.pk', 'rb') as f:
        indices = pickle.load(f)

    # FOR TESTING
    # indices = {'Train': ['CC00050XX01_7201', 'CC00051XX02_7702'],
    #            'Test': ['CC00050XX01_7201', 'CC00051XX02_7702'],
    #            'Val': ['CC00050XX01_7201', 'CC00051XX02_7702']}
    ###############################################

    data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere=hemisphere)

    train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(
        data_folder,
        files_ending,
        data_type,
        target_class,
        task,
        REPROCESS,
        local_features,
        global_features,
        indices,
        batch_size,
        num_workers=2,
        data_nativeness=data_nativeness,
        data_compression=data_compression,
        hemisphere=hemisphere
    )

    if len(local_features) > 0:
        numb_local_features = train_dataset[0].x.size(1)
    else:
        numb_local_features = 0
    numb_global_features = len(global_features)

    # 7. Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(numb_local_features, numb_global_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    print(f'number of param: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    #################################################
    ############# EXPERIMENT LOGGING ################
    #################################################
    writer = None
    results_folder = None
    if recording:

        # Tensorboard writer.
        writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)

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
            config_file.write('Data res - ' + data_compression + '\n')
            config_file.write('Data type - ' + data_type + '\n')
            config_file.write('Data nativeness - ' + data_nativeness + '\n')
            # config_file.write('Additional comments - With rotate transforms' + '\n')

        with open(results_folder + '/results.csv', 'w', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label', 'Error'])

    #################################################
    #################################################

    best_val_loss = 999

    # MAIN TRAINING LOOP
    for epoch in range(1, numb_epochs + 1):
        start = time.time()
        train(model, train_loader, epoch, device,
              optimizer, scheduler, writer)

        val_mse, val_l1 = test_regression(model, val_loader,
                                          indices['Val'], device,
                                          recording, results_folder,
                                          epoch=epoch)

        if recording:
            writer.add_scalar('Loss/val_mse', val_mse, epoch)
            writer.add_scalar('Loss/val_l1', val_l1, epoch)

            end = time.time()
            print('Time: ' + str(end - start))
            if val_l1 < best_val_loss:
                best_val_loss = val_l1
                torch.save(model.state_dict(), model_dir + '/model_best.pt')
                print('Saving Model'.center(60, '-'))
            writer.add_scalar('Time/epoch', end - start, epoch)

    test_regression(model, test_loader, indices['Test'], device, recording, results_folder, val=False)

    if recording:
        # save the last model
        torch.save(model.state_dict(), model_dir + '/model_last.pt')

        # Eval best model on test
        model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Best model!'])

        test_regression(model, test_loader, indices['Test'], device, recording, results_folder, val=False)
