########################################
# Imports for Model
########################################
from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, Dropout, BatchNorm1d
from torch.optim import Adam, lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
########################################
# End Imports
########################################
variances = [i for i in range(30)]


"""# Dataset & Preprocessing"""


class ImageSegmentationDataset(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, selected_ids, id_ages, smoothen=None, edgen=False):
        if smoothen is None:
            smoothen = 0
        print("Initialising Dataset")
        self.ids = selected_ids
        if edgen:  # resample_image(dts[0][0], [3, 3, 3], [60, 60, 50])
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.DiscreteGaussian(resample_image(sitk.ReadImage(f"gm_volume3d/sub-{ID[0]}_ses-{ID[1]}_T2w_graymatter.nii.gz", sitk.sitkFloat32), [3, 3, 3], [60, 60, 50]), smoothen)))).unsqueeze(0) for ID in self.ids]
        else:
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.DiscreteGaussian(resample_image(sitk.ReadImage(f"gm_volume3d/sub-{ID[0]}_ses-{ID[1]}_T2w_graymatter.nii.gz", sitk.sitkFloat32), [3, 3, 3], [60, 60, 50]), smoothen))).unsqueeze(0) for ID in self.ids]

        # self.samples = [(sitk.DiscreteGaussian(sitk.ReadImage(f"{data_dir}/greymatter/wc1sub-{ID}_T1w.nii.gz", sitk.sitkFloat32), smoothen)) for ID in self.ids]
        self.targets = torch.tensor(id_ages, dtype=torch.float).view((-1, 1))
        print("Initialisation complete")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.samples[item], self.targets[item]


#
# class ImageSegmentationDataset(Dataset):
#     """Dataset for image segmentation."""
#
#     def __init__(self, file_list_img, ages, img_spacing, img_size, smooth, edge=False, sharpen=False):
#         self.samples = []
#         self.img_names = []
#         self.ages = ages
#
#         for idx, _ in enumerate(tqdm(range(len(file_list_img)), desc='Loading Data')):
#
#             # 1. Get image
#             img_path = file_list_img[idx]
#             img = sitk.Cast(sitk.ReadImage(img_path), sitk.sitkFloat32)
#
#             # 2. Pre-process the image
#             img = resample_image(img, img_spacing, img_size, is_label=False)
#
#             if smooth in variances:
#                 img = sitk.DiscreteGaussian(img, smooth)
#
#             elif smooth == 'diffusion':
#                 img = sitk.GradientAnisotropicDiffusion(img)
#
#             if edge:
#                 img = sitk.SobelEdgeDetection(img)
#
#             if sharpen:
#                 img_sm = sitk.DiscreteGaussian(img, 1)
#                 a = img - img_sm
#                 img = img + a * 2
#
#                 # 3. Append to the samples list
#             self.samples.append(img)
#             self.img_names.append(os.path.basename(img_path))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, item):
#         sample = self.samples[item]
#         image = torch.from_numpy(sitk.GetArrayFromImage(sample)).unsqueeze(0)
#
#         return image
#
#     def get_sample(self, item):
#         return self.samples[item]
#
#     def get_img_name(self, item):
#         return self.img_names[item]
#
#     def get_seg_name(self, item):
#         return self.seg_names[item]
#
#     def split_data(self):
#
#         numpy_samples = []
#
#         for sample in self.samples:
#             numpy_samples.append(sitk.GetArrayFromImage(sample).flatten())
#
#         numpy_samples = np.array(numpy_samples)
#
#         X_train, X_test, y_train, y_test = train_test_split(numpy_samples, self.ages, test_size=0.50, random_state=42)
#         return X_train, X_test, y_train, y_test
#
#     def get_data(self):
#         numpy_samples = []
#
#         for sample in self.samples:
#             numpy_samples.append(sitk.GetArrayFromImage(sample).flatten())
#
#         return np.array(numpy_samples), self.ages


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




"""# Model Architecture"""

DO_PRINT = True


class PrintTensor(nn.Module):
    # Custom helper layer to display shape of batch input/output
    def __init__(self, name="", do_print=DO_PRINT):
        super(PrintTensor, self).__init__()
        self.name=name
        self.do_print = do_print

    def forward(self, x):
        if self.do_print:
            print(f"{self.name}: {x.size()}")
        return x


class Part3(Module):
    """
    Neural Network for part 3.
    """

    def __init__(self, feats, dropout_p):
        super(Part3, self).__init__()
        self.model = Sequential(
            # 50, 60, 60
            Conv3d(1, feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(feats),
            ReLU(),
            Conv3d(feats, feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(feats),
            ReLU(),
            Conv3d(feats, 2*feats, padding=0, kernel_size=2, stride=2, bias=True),
            Dropout(p=dropout_p),

            # 23, 28, 28
            Conv3d(2*feats, 2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2*feats),
            ReLU(),
            Conv3d(2*feats, 2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2*feats),
            ReLU(),
            Conv3d(2*feats, 2*2*feats, padding=0, kernel_size=2, stride=2, bias=True),

            # 9, 12, 12
            Conv3d(2*2*feats, 2*2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2*2*feats),
            ReLU(),
            Conv3d(2*2*feats, 2*2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            BatchNorm3d(2*2*feats),
            ReLU(),
            Conv3d(2*2*feats, 2*2*2*feats, padding=0, kernel_size=1, stride=1, bias=True),
            Dropout(p=dropout_p),

            # 5, 8, 8
            Conv3d(2*2*2*feats, 2*2*2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            # 3, 6, 6
            BatchNorm3d(2*2*2*feats),
            ReLU(),
            Conv3d(2*2*2*feats, 2*2*2*feats, padding=0, kernel_size=3, stride=1, bias=True),
            # 1, 4, 4
            BatchNorm3d(2*2*2*feats),
            ReLU(),
            Conv3d(2*2*2*feats, 2*2*2*2*feats, padding=0, kernel_size=(1, 2, 2), stride=1, bias=True),
            Dropout(p=dropout_p),
            #  1, 3, 3
            Flatten(start_dim=1), # Output: 1
            Linear(2*2*2*2*feats*(1*3*3), 1),
            )

    def forward(self, x):
        return self.model(x)