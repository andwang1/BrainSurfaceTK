import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import pickle
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.pointnet.src.utils import get_id, save_to_log, get_comment, get_data_path, data, get_grid_search_local_features
from models.pointnet.src.models.pointnet2_segmentation import Net
from models.pointnet.main.pointnet2_segmentation import train, test, perform_final_testing

# Global variables
all_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17])
num_points_dict = {'original': 32492, '50': 16247, '90': None}
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
    data_compression = "20k"
    data_type = 'white'
    hemisphere = 'left'

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
    target_class = 'gender'
    task = 'segmentation'
    REPROCESS = True
    ################################################

    # 2. Get the data splits indices
    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)

    # 4. Get experiment description
    comment = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                          lr, batch_size, local_features, global_features, target_class)

    print('=' * 50 + '\n' + '=' * 50)
    print(comment)
    print('=' * 50 + '\n' + '=' * 50)

    # 5. Perform data processing
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

    # 6. Getting the number of features to adapt the architecture
    try:
        num_local_features = train_dataset[0].x.size(1)
    except:
        num_local_features = 0
    print(f'Unique labels found: {num_labels}')

    if not torch.cuda.is_available():
        print('You are running on a CPU.')

    # 7. Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_labels, num_local_features, num_global_features=None).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    id = '0'
    if recording:
        # 9. Save to log_record.txt
        log_descr = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                                lr, batch_size, local_features, global_features, target_class,
                                log_descr=True)

        save_to_log(log_descr, prefix=experiment_name)

        id = str(int(get_id(prefix=experiment_name)) - 1)

        writer = SummaryWriter(PATH_TO_POINTNET + f'new_runs/{experiment_name}ID' + id)
        writer.add_text(f'{experiment_name} ID #{id}', comment)

    best_val_acc = 0
    best_val_iou = 0
    best_model_acc = 0
    best_model_iou = 0

    # 10. ====== TRAINING LOOP ======
    for epoch in range(1, 3):

        # 1. Start recording time
        start = time.time()

        # 2. Make a training step
        train(model, train_loader, epoch, device, optimizer, num_labels, writer, recording=recording)

        if recording:
            writer.add_scalar('Training Time/epoch', time.time() - start, epoch)

        # 3. Validate the performance after each epoch
        loss, acc, iou, mean_iou = test(model, val_loader, comment + 'val' + str(epoch), device, num_labels, writer, epoch=epoch, id=id,
                                        experiment_name=experiment_name, recording=recording)
        print('Epoch: {:02d}, Val Loss/nll: {}, Val Acc: {:.4f}'.format(epoch, loss, acc))

        scheduler.step()

        # 4. Record valiation metrics in Tensorboard
        if recording:

            # By Accuracy
            if acc > best_val_acc:
                best_val_acc = acc
                best_model_acc = epoch
                torch.save(model.state_dict(),
                           PATH_TO_POINTNET + f'experiment_data/new/{experiment_name}-{id}/' + 'best_acc_model' + '.pt')

            # By Mean IoU
            if mean_iou > best_val_iou:
                best_val_iou = mean_iou
                best_model_iou = epoch
                torch.save(model.state_dict(),
                           PATH_TO_POINTNET + f'experiment_data/new/{experiment_name}-{id}/' + 'best_iou_model' + '.pt')

            writer.add_scalar('Loss/val_nll', loss, epoch)
            writer.add_scalar('Accuracy/val', acc, epoch)
            for label, value in enumerate(iou):
                writer.add_scalar('IoU{}/validation'.format(label), value, epoch)
                print('\t\tValidation Label {}: {}'.format(label, value))

        print('=' * 60)

    # save the last model
    torch.save(model.state_dict(), PATH_TO_POINTNET + f'experiment_data/new/{experiment_name}-{id}/' + 'last_model' + '.pt')

    loss_acc, acc_acc, iou_acc, mean_iou_acc, loss_iou, acc_iou, iou_iou, mean_iou_iou = perform_final_testing(model,
                                                                                                               writer,
                                                                                                               test_loader,
                                                                                                               experiment_name,
                                                                                                               comment,
                                                                                                               id,
                                                                                                               num_labels,
                                                                                                               device,
                                                                                                               best_model_acc,
                                                                                                               best_model_iou,
                                                                                                               recording=recording)