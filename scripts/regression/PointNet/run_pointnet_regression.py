import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)

import os
import time
import pickle
import csv
import datetime as datetime

import torch
from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from models.pointnet.src.models.pointnet2_regression_v2 import Net
from models.pointnet.src.data_loader import OurDataset
from models.pointnet.main.pointnet2 import train, test_regression

from models.pointnet.src.utils import get_id, save_to_log, get_comment, get_data_path, data

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..') + '/'
PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'pointnet') + '/'

if __name__ == '__main__':

    num_workers = 2
    local_features = ['corr_thickness', 'curvature', 'sulc']
    global_features = []

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = True
    REPROCESS = True

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'pial'
    hemisphere = 'both'

    additional_comment = ''

    experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{hemisphere}_{additional_comment}'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################


    # 1. Model Parameters
    ################################################
    lr = 0.001
    batch_size = 2
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'scan_age'
    task = 'regression'
    numb_epochs = 200
    number_of_points = 10000
    comment = 'comment'
    ################################################


    ########## INDICES FOR DATA SPLIT #############
    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)
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
            config_file.write('Data res - ' + data + '\n')
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
    for epoch in range(1, numb_epochs+1):
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

            print('Epoch: {:03d}, Test loss l1: {:.4f}'.format(epoch, val_l1))
            end = time.time()
            print('Time: ' + str(end - start))
            if val_l1 < best_val_loss:
                best_val_loss = val_l1
                torch.save(model.state_dict(), model_dir + '/model_best.pt')
                print('Saving Model'.center(60, '-'))
            writer.add_scalar('Time/epoch', end - start, epoch)

    test_regression(model, test_loader, indices['Test'], device, results_folder, val=False)

    if recording:
        # save the last model
        torch.save(model.state_dict(), model_dir + '/model_last.pt')

        # Eval best model on test
        model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Best model!'])

        test_regression(model, test_loader, indices['Test'], device, results_folder, val=False)





    # ############# MODEL PARAMETERS ################
    # lr = 0.001
    # batch_size = 4
    # num_workers = 4
    # # ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc'] + ['weight']
    # local_features = ['corr_thickness', 'curvature', 'sulc']
    # # local_features = ['sulc']
    # global_features = []
    # target_class = 'scan_age'
    # # target_class = 'birth_age'
    # task = 'regression'
    # number_of_points = 500  # 3251# 12000  # 16247
    #
    # reprocess = False
    # ###############################################

    ############# DATA INFORMATION ################
    # data = "reducedto_05k"
    # data_ending = "05k.vtk"
    # type_data_surf = "pial"
    # type_data_part = "merged"
    #
    # # folder in data/stored for pre-processed data.
    # stored = target_class + '/' + type_data_surf + '/' + data + '/' + str(local_features + global_features) + '/' + type_data_part
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/' + stored)
    # data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/' + \
    #               type_data_part + '/' + data + '/' + type_data_surf + '/vtk'
    #
    # # sub-CC00466AN13_ses-138400_right_pial_reduce90.vtk
    # files_ending = '_' + type_data_part + '_' + type_data_surf + '_' + data_ending
    # ###############################################
    #
    # ########## INDICES FOR DATA SPLIT #############
    # with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
    #     indices = pickle.load(f)
    # ###############################################
    #
    # ####### Keeping track of the resulst ##########
    # comment = 'TEST_Sphere_scan_age_90' + str(datetime.datetime.now()) \
    #           + "__LR__" + str(lr) \
    #           + "__BATCH_" + str(batch_size) \
    #           + "__local_features__" + str(local_features) \
    #           + "__glogal_features__" + str(global_features) \
    #           + "__number_of_points__" + str(number_of_points) \
    #           + "__" + data + "__" + type_data_surf + '__no_rotate'
    #
    # results_folder = 'runs/' + task + '/' + comment + '/results'
    # model_dir = 'runs/' + task + '/' + comment + '/models'
    #
    # if not osp.exists(results_folder):
    #     os.makedirs(results_folder)
    #
    # if not osp.exists(model_dir):
    #     os.makedirs(model_dir)
    #
    # with open(results_folder + '/configuration.txt', 'w', newline='') as config_file:
    #     config_file.write('Learning rate - ' + str(lr) + '\n')
    #     config_file.write('Batch size - ' + str(batch_size) + '\n')
    #     config_file.write('Local features - ' + str(local_features) + '\n')
    #     config_file.write('Global feature - ' + str(global_features) + '\n')
    #     config_file.write('Number of points - ' + str(number_of_points) + '\n')
    #     config_file.write('Data res - ' + data + '\n')
    #     config_file.write('Data type - ' + type_data_surf + '\n')
    #     config_file.write('Additional comments - With rotate transforms' + '\n')
    #
    # with open(results_folder + '/results.csv', 'w', newline='') as results_file:
    #     result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label', 'Error'])
    #
    # # Tensorboard writer.
    # writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)
    # ###############################################
    #
    # # DEFINE TRANSFORMS HERE.
    # transform = T.Compose([
    #     T.FixedPoints(number_of_points),
    #     T.RandomRotate(360, axis=0),
    #     T.RandomRotate(360, axis=1),
    #     T.RandomRotate(360, axis=2)
    # ])
    #
    # # TRANSFORMS DONE BEFORE SAVING THE DATA IF THE DATA IS NOT YET PROCESSED.
    # pre_transform = T.NormalizeScale()
    #
    # # Creating datasets and dataloaders for train/test/val.
    # train_dataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform, val=False,
    #                            target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
    #                            local_features=local_features, global_feature=global_features,
    #                            indices=indices['Train'], data_folder=data_folder)
    #
    # test_dataset = OurDataset(path, train=False, transform=None, pre_transform=pre_transform, val=False,
    #                           target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
    #                           local_features=local_features, global_feature=global_features,
    #                           indices=indices['Test'], data_folder=data_folder)
    #
    # val_dataset = OurDataset(path, train=False, transform=None, pre_transform=pre_transform, val=True,
    #                          target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
    #                          local_features=local_features, global_feature=global_features,
    #                          indices=indices['Val'], data_folder=data_folder)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    #
    # # Getting the number of features to adapt the architecture
    # if len(local_features) > 0:
    #     numb_local_features = train_dataset[0].x.size(1)
    # else:
    #     numb_local_features = 0
    # numb_global_features = len(global_features)
    #
    # if not torch.cuda.is_available():
    #     print('YOU ARE RUNNING ON A CPU!!!!')
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Net(numb_local_features, numb_global_features).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.985)
    #
    # print(f'number of param: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    #
    # best_val_loss = 999
    #
    # # MAIN TRAINING LOOP
    # for epoch in range(1, 201):
    #     start = time.time()
    #     train(model, train_loader, epoch, device, optimizer, writer)
    #
    #     val_mse, val_l1 = test_regression(model, val_loader, indices['Val'], device, results_folder, epoch=epoch)
    #
    #     scheduler.step()
    #
    #     writer.add_scalar('Loss/val_mse', val_mse, epoch)
    #     writer.add_scalar('Loss/val_l1', val_l1, epoch)
    #
    #     print('Epoch: {:03d}, Test loss l1: {:.4f}'.format(epoch, val_l1))
    #     end = time.time()
    #     print('Time: ' + str(end - start))
    #     if val_l1 < best_val_loss:
    #         best_val_loss = val_l1
    #         torch.save(model.state_dict(), model_dir + '/model_best.pt')
    #         print('Saving Model'.center(60, '-'))
    #     writer.add_scalar('Time/epoch', end - start, epoch)
    #
    # test_regression(model, test_loader, indices['Test'], device, results_folder, val=False)
    #
    # # save the last model
    # torch.save(model.state_dict(), model_dir + '/model_last.pt')
    #
    # # Eval best model on test
    # model.load_state_dict(torch.load(model_dir + '/model_best.pt'))
    #
    # with open(results_folder + '/results.csv', 'a', newline='') as results_file:
    #     result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #     result_writer.writerow(['Best model!'])
    #
    # test_regression(model, test_loader, indices['Test'], device, results_folder, val=False)
