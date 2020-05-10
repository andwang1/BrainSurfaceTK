import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import torch
from models.volume3d.utils.utils import read_meta, clean_data, split_data, get_ids_and_ages, plot_preds
import os.path as osp
from models.volume3d.main.train_validate import train_validate, save_to_log
from models.volume3d.main.train_test import train_test, save_to_log_test
from torch.utils.tensorboard import SummaryWriter
from models.volume3d.main.main import create_subject_folder


cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'
PATH_TO_VOLUME3D = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'volume3d') + '/'


if __name__ == '__main__':


    additional_comment = ''

    # 1. What are you predicting?
    categories = {'gender': 3, 'birth_age': 4, 'weight': 5, 'scan_age': 7, 'scan_num': 8}
    meta_column_idx = categories['scan_age']

    # 2. Read the data and clean it
    meta_data = read_meta()

    ## 3. Get a list of ids and ages (labels)
    # ids, ages = get_ids_and_ages(meta_data, meta_column_idx)

    # 4. Set the parameters for the data pre-processing and split
    ################################
    ################################

    spacing = [3, 3, 3]
    image_size = [60, 60, 50]
    smoothen = 8
    edgen = False
    test_size = 0.09
    val_size = 0.1
    random_state = 42
    REPROCESS = False

    ################################
    ################################


    # 4. Create subject folder
    fn, counter = create_subject_folder()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')

    # 5. Split the data
    dataset_train, dataset_val, dataset_test = split_data(meta_data,
                                                          meta_column_idx,
                                                          spacing,
                                                          image_size,
                                                          smoothen,
                                                          edgen,
                                                          val_size,
                                                          test_size,
                                                          path=path,
                                                          reprocess=REPROCESS)

    # 6. Create CNN Model parameters
    ################################
    ################################
    ################################

    USE_GPU = True
    dtype = torch.float32
    num_of_parameters_multiplier = 10
    num_epochs = 3
    lr = 0.006882801723742766
    gamma = 0.97958263796472
    batch_size = 32
    dropout_p = 0.5
    scheduler_frequency = 3

    ################################
    ################################
    ################################

    # 6. Create tensorboard writer
    writer = SummaryWriter(PATH_TO_VOLUME3D + f'tensorboard_runs/Subject {additional_comment} {counter}')

    # 7. Run TRAINING + VALIDATION after every N epochs
    model, params, final_MAE = train_validate(lr, num_of_parameters_multiplier, num_epochs,
                                              gamma, batch_size,
                                              dropout_p, dataset_train,
                                              dataset_val, fn, counter,
                                              scheduler_frequency,
                                              writer=writer)

    # 8. Save the results
    save_to_log(model, params,
                fn, final_MAE,
                num_epochs,
                batch_size,
                lr, num_of_parameters_multiplier,
                gamma, smoothen,
                edgen, dropout_p,
                spacing, image_size,
                scheduler_frequency)


    """# Full Train & Final Test"""

    # 2. Create TEST folder
    fn, counter = create_subject_folder(test=True)

    # 3. Run TRAINING + TESTING after every N epochs
    model, params, score, train_loader, test_loader = train_test(lr, num_of_parameters_multiplier, num_epochs, gamma,
                                                                 batch_size, dropout_p,
                                                                 dataset_train, dataset_test,
                                                                 fn, counter,
                                                                 scheduler_frequency,
                                                                 writer=writer)

    # 4. Record the TEST results
    save_to_log_test(model, params, fn, score, num_epochs, batch_size,
                     lr, num_of_parameters_multiplier, gamma, smoothen, edgen, dropout_p, spacing,
                     image_size, scheduler_frequency)


    # 5. Perform the final testing
    model.eval()
    pred_ages = []
    actual_ages = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)

            pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
            actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

    plot_preds(pred_ages, actual_ages, writer, num_epochs, test=True)
