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
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from src.data_loader import OurDataset

from src.models.pointnet2_regression import Net


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

    writer.add_scalar('Loss/train_mse', loss_train / len(train_loader), epoch)


def test_regression(loader, indices, results_folder, val=True, epoch=0):

    model.eval()
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
                print(str(pred.t().item()).center(20, ' '), str(data.y[:, 0].item()).center(20, ' '), indices[idx])
                result_writer.writerow([indices[idx][:11], indices[idx][12:],
                                        str(pred.t().item()), str(data.y[:, 0].item()),
                                        str(abs(pred.t().item() - data.y[:, 0].item()))])
                loss_test_mse = F.mse_loss(pred, data.y[:, 0])
                loss_test_l1 = F.l1_loss(pred, data.y[:, 0])
                mse += loss_test_mse.item()
                l1 += loss_test_l1.item()
        if val:
            result_writer.writerow(['Epoch average error:', str(l1 / len(loader))])
        else:
            result_writer.writerow(['Test average error:', str(l1 / len(loader))])

    return mse / len(loader), l1 / len(loader)


if __name__ == '__main__':

    # Model Parameters
    lr = 0.001
    batch_size = 16
    num_workers = 4
    # ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc'] + ['weight']
   # local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    local_features = ['sulc']
    global_features = []
    target_class = 'scan_age'
    #target_class = 'birth_age'
    task = 'regression'
    number_of_points = 5000  #3251# 12000  # 16247

    # For quick tests
    # indices = {'Train': ['CC00050XX01_7201', 'CC00050XX01_7201'],
    #            'Test': ['CC00050XX01_7201', 'CC00050XX01_7201'],
    #            'Val': ['CC00050XX01_7201', 'CC00050XX01_7201']}

    reprocess = False

    #inflated  midthickness  pial  sphere  veryinflated  white
    # NATIVE: inflated  midthickness	pial  very_inflated  white
    #data = "reduced_50"
    #data_ending = "reduce50.vtk"
    #type_data = "inflated"

    data = "reducedto_05k"
    data_ending = "05k.vtk"
    #type_data = "pial"
    type_data = "sphere"
    native = "merged"#"surface_native" #surface_fsavg32k

    # folder in data/stored for data.
    stored = target_class + '/' + type_data + '/' + data + '/' + str(local_features + global_features) + '/' + native

    # ALIGNED OLD
    # data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/" + native + "/" + data \
    #               + "/vtk/" + type_data

   # Native OLD
   # data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/" + native + "/" + data \
   #               + "/" + type_data + "/vtk"

    data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/'+\
                  native+'/'+data+'/' + type_data + '/vtk'

    print(data_folder)
   # sub-CC00466AN13_ses-138400_right_pial_reduce90.vtk
   #  files_ending = "_hemi-L_" + type_data + "_" + data_ending
   # files_ending = "_left_" + type_data + "_" + data_ending
    files_ending = '_'+ native + '_' + type_data + '_' + data_ending

    # From quick local test
    # data_folder = "/home/vital/Group Project/deepl_brain_surfaces/random"

    with open('src/names.pk', 'rb') as f:
        indices = pickle.load(f)

    # TESTING PURPOSES
    # indices['Train'] = indices['Train'][:2]

    comment = 'Sphere_scan_age_90' + str(datetime.datetime.now()) \
            + "__LR__" + str(lr) \
            + "__BATCH_" + str(batch_size) \
            + "__local_features__" + str(local_features)\
            + "__glogal_features__" + str(global_features) \
            + "__number_of_points__" + str(number_of_points)\
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
        config_file.write('Additional comments - With rotate transforms' + '\n')

    with open(results_folder + '/results.csv', 'w', newline='') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label', 'Error'])

    # Tensorboard writer.
    writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/' + stored)

    # DEFINE TRANSFORMS HERE.
    transform = T.Compose([
        # T.FixedPoints(number_of_points),
        T.RandomRotate(360, axis=0),
        T.RandomRotate(360, axis=1),
        T.RandomRotate(360, axis=2)
    ])

    # TRANSFORMS DONE BEFORE SAVING THE DATA IF THE DATA IS NOT YET PROCESSED.
    pre_transform = T.NormalizeScale()

    # Creating datasets and dataloaders for train/test/val.
    train_dataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform, val=False,
                               target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
                               local_features=local_features, global_feature=global_features,
                               indices=indices['Train'], data_folder=data_folder)

    test_dataset = OurDataset(path, train=False, transform=None, pre_transform=pre_transform, val=False,
                              target_class=target_class, task=task, reprocess=reprocess, files_ending=files_ending,
                              local_features=local_features, global_feature=global_features,
                              indices=indices['Test'], data_folder=data_folder)

    val_dataset = OurDataset(path, train=False, transform=None, pre_transform=pre_transform, val=True,
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

    best_val_loss = 999

    # MAIN TRAINING LOOP
    for epoch in range(1, 201):
        start = time.time()
        train(epoch)
        val_mse, val_l1 = test_regression(val_loader, indices['Val'], results_folder, epoch=epoch)
        
        scheduler.step()

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

    test_regression(test_loader, indices['Test'], results_folder, val=False)

    # save the last model
    torch.save(model.state_dict(), model_dir + '/model_last.pt')

    # Eval best model on test
    model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

    with open(results_folder + '/results.csv', 'a', newline='') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['Best model!'])

    test_regression(test_loader, indices['Test'], results_folder, val=False)
