import csv
from sklearn.model_selection import train_test_split
import os
from ipywidgets import interact, fixed
from IPython.display import display
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.models import ImageSegmentationDataset
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pickle


def plot_to_tensorboard(writer, fig, name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(f'{name}', img)
    plt.close(fig)



def zero_mean_unit_var(image, mask):
    """Normalizes an image to zero mean and unit variance."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = sitk.GetArrayFromImage(mask)

    mean = np.mean(img_array[msk_array>0])
    std = np.std(img_array[msk_array>0])

    if std > 0:
        img_array = (img_array - mean) / std
        img_array[msk_array==0] = 0

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)



path = 'combined_exist.tsv'
data_dir = './gm_volume3d/'

def read_meta(path=path):
    '''Correctly reads a .tsv file into a numpy array'''

    with open(path) as fd:
        rd = csv.reader(fd, delimiter=",", quotechar='"')
        data = []
        for idx, row in enumerate(rd):
            if idx == 0:
                continue
            data.append(row)

    data = np.array(data)
    return data


def clean_data(meta_data):
    '''Cleans the meta_data. Removes rows in the data that we don't have files from.
    :param meta_data: meta data.
    :return data: cleaned version of meta data'''

    missing_idx = []

    for idx, patient_id in enumerate(meta_data[:, 1]):
        file_path = get_file_path(patient_id, meta_data[idx, 2])
        if not os.path.isfile(file_path):
            missing_idx.append(idx)

    # for idx in missing_idx:
    meta_data = np.delete(meta_data, missing_idx, 0)

    return meta_data


def get_file_path(patient_id, session_id):
    # sub-CC00050XX01_ses-7201_T2w_graymatter.nii.gz
    file_name = "sub-" + patient_id +"_ses-" + session_id + '_T2w_graymatter.nii.gz'
    file_path = data_dir + file_name

    return file_path


# Metadata categories
test_size = 0.09
val_size = 0.1
random_state = 42


def split_data(meta_data, meta_column_idx, ids, ages, spacing, image_size, smoothen, edgen, val_size, test_size, random_state=42, path='./', reprocess=True):
    '''
    Splits the data

    :param meta_data:
    :param meta_column_idx: the column index of the label
    :param ids:
    :param ages:
    :param val_size:
    :param test_size:
    :param random_state:
    :return:
    '''

    _, bins = np.histogram(meta_data[:, meta_column_idx].astype(float), bins='doane')
    y_binned = np.digitize(meta_data[:, meta_column_idx].astype(float), bins)

    X_train, X_test, y_train, y_test = train_test_split(ids, ages,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y_binned)

    if val_size > 0:
        _, bins = np.histogram(np.array(y_train).astype(float), bins='doane')
        y_binned = np.digitize(np.array(y_train).astype(float), bins)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=val_size,
                                                          random_state=random_state,
                                                          stratify=y_binned)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '../')

    # ImageSegmentationDataset
    if os.path.exists(path + 'dataset_train.pkl') and reprocess == False:

        with open(path + 'dataset_train.pkl', 'rb') as file:
            dataset_train = pickle.load(file)
        with open(path + 'dataset_val.pkl', 'rb') as file:
            dataset_val = pickle.load(file)
        with open(path + 'dataset_test.pkl', 'rb') as file:
            dataset_test = pickle.load(file)
    else:
        dataset_train = ImageSegmentationDataset(path, X_train, y_train, spacing, image_size, smoothen, edgen)
        dataset_val = ImageSegmentationDataset(path, X_val, y_val, spacing, image_size, smoothen, edgen)
        dataset_test = ImageSegmentationDataset(path, X_test, y_test, spacing, image_size, smoothen, edgen)

        with open('dataset_train.pkl', 'wb') as file:
            pickle.dump(dataset_train, file)
        with open('dataset_val.pkl', 'wb') as file:
            pickle.dump(dataset_val, file)
        with open('dataset_test.pkl', 'wb') as file:
            pickle.dump(dataset_test, file)

    return dataset_train, dataset_val, dataset_test


def get_ids_and_ages(meta_data, meta_column_idx):
    # 3. Iterate through all patient ids
    ids = []
    ages = []
    for idx, patient_id in enumerate(meta_data[:, 1]):
        session_id = meta_data[idx, 2]
        file_path = get_file_path(patient_id, session_id)
        if os.path.isfile(file_path):
            ids.append((patient_id, session_id))
            ages.append(float(meta_data[idx, meta_column_idx]))

    return ids, ages



# Calculate parameters low and high from window and level
def wl_to_lh(window, level):
    low = level - window / 2
    high = level + window / 2
    return low, high


def display_image(path, img, img_idx, x=None, y=None, z=None, window=None, level=None, colormap='gray', crosshair=False):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)

    # Get image dimensions in millimetres
    size = img.GetSize()
    spacing = img.GetSpacing()
    width = size[0] * spacing[0]
    height = size[1] * spacing[1]
    depth = size[2] * spacing[2]

    if x is None:
        x = np.floor(size[0] / 2).astype(int)
    if y is None:
        y = np.floor(size[1] / 2).astype(int)
    if z is None:
        z = np.floor(size[2] / 2).astype(int)

    if window is None:
        window = np.max(img_array) - np.min(img_array)

    if level is None:
        level = window / 2 + np.min(img_array)

    low, high = wl_to_lh(window, level)

    # Display the orthogonal slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    ax1.imshow(img_array[z, :, :], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(img_array[:, y, :], origin='lower', cmap=colormap, clim=(low, high), extent=(0, width, 0, depth))
    ax3.imshow(img_array[:, :, x], origin='lower', cmap=colormap, clim=(low, high), extent=(0, height, 0, depth))

    # Additionally display crosshairs
    if crosshair:
        ax1.axhline(y * spacing[1], lw=1)
        ax1.axvline(x * spacing[0], lw=1)
        ax2.axhline(z * spacing[2], lw=1)
        ax2.axvline(x * spacing[0], lw=1)
        ax3.axhline(z * spacing[2], lw=1)
        ax3.axvline(y * spacing[1], lw=1)

    plt.show()
    plt.savefig(path + f'preprocessed_exemplar{img_idx}.png')
    plt.close()


def interactive_view(img):
    size = img.GetSize()
    img_array = sitk.GetArrayFromImage(img)
    interact(display_image, img=fixed(img),
             x=(0, size[0] - 1),
             y=(0, size[1] - 1),
             z=(0, size[2] - 1),
             window=(0, np.max(img_array) - np.min(img_array)),
             level=(np.min(img_array), np.max(img_array)));