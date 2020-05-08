import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_meta, clean_data, split_data, get_ids_and_ages, plot_preds
import os.path as osp
from main.train_validate import train_validate, save_to_log
from volume3d.main.train_test import train_test, save_to_log_test
from torch.utils.tensorboard import SummaryWriter


cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

sns.set(style='darkgrid')


def plot_to_image(figure):

    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def create_subject_folder(test=False):
    '''
    Creates a folder according to the number of previous tests performed
    '''

    additional_comment = 'final'


    name = 'Subject'
    if test:
        name = 'Test'

    print(f"Creating {name} Folder")
    number_here = 0
    while True:

        fn = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{name}_{additional_comment}_{number_here}')

        if not os.path.exists(fn):
            print(f"Making {number_here}")
            os.makedirs(fn)

            with open(f'{fn}/log.txt', 'w+') as log:
                log.write('\n')
            break

        else:
            print(f"{name}_{number_here} exists")
            number_here += 1

    print("Created Subject Folder")

    return fn, number_here




if __name__ == '__main__':

    additional_comment = 'final'

    # 1. What are you predicting?
    categories = {'gender': 3, 'birth_age': 4, 'weight': 5, 'scan_age': 7, 'scan_num': 8}
    meta_column_idx = categories['scan_age']

    # 2. Read the data and clean it
    meta_data = read_meta()

    ## 3. Get a list of ids and ages (labels)
    # ids, ages = get_ids_and_ages(meta_data, meta_column_idx)

    # 4. Set the parameters for the data pre-processing and split
    spacing = [3, 3, 3]
    image_size = [60, 60, 50]
    smoothen = 8
    edgen = False
    test_size = 0.09
    val_size = 0.1
    random_state = 42
    REPROCESS = False

    features_list = [15, 10, 9, 8, 6, 5]

    for feats in features_list:
        for scheduler_frequency in [1, 3, 5]:

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
            USE_GPU = True
            dtype = torch.float32
            feats = feats
            num_epochs = 1000
            lr = 0.006882801723742766
            gamma = 0.97958263796472
            batch_size = 32
            dropout_p = 0.5
            scheduler_frequency = scheduler_frequency

            # 6. Create tensorboard writer
            writer = SummaryWriter(f'runs/Subject {additional_comment} {counter}')

            # 7. Run TRAINING + VALIDATION after every N epochs
            model, params, final_MAE = train_validate(lr, feats, num_epochs,
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
                        lr, feats,
                        gamma, smoothen,
                        edgen, dropout_p,
                        spacing, image_size,
                        scheduler_frequency)


            """# Full Train & Final Test"""

            # 2. Create TEST folder
            fn, counter = create_subject_folder(test=True)

            # 3. Run TRAINING + TESTING after every N epochs
            model, params, score, train_loader, test_loader = train_test(lr, feats, num_epochs, gamma,
                                                                         batch_size, dropout_p,
                                                                         dataset_train, dataset_test,
                                                                         fn, counter,
                                                                         scheduler_frequency,
                                                                         writer=writer)

            # 4. Record the TEST results
            save_to_log_test(model, params, fn, score, num_epochs, batch_size,
                             lr, feats, gamma, smoothen, edgen, dropout_p, spacing,
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
