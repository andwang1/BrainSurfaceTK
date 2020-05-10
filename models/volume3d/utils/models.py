########################################
# Imports for Model
########################################
from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, Dropout, BatchNorm1d
import numpy as np
from torch import nn
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
########################################
# End Imports
########################################

variances = [i for i in range(30)]
PATH_TO_DATA = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex2/deepl_brain_surfaces/'


class ImageSegmentationDataset(Dataset):
    """Dataset for image segmentation.
        selected_ids: tuples with patient id and session id [ (,) ,  (,) ,  ... ]
        id_ages: labels
        """

    def __init__(self, path, selected_ids, id_ages, spacing=[3, 3, 3], image_size=[60, 60, 50], smoothen=None, edgen=False):
        '''
        :param path: path to saving folder
        :param selected_ids: which patients to process
        :param id_ages: labels
        :param spacing: image spacing
        :param image_size: image size
        :param smoothen: apply smoothening filter (True/False)
        :param edgen: apply edgening filter (True/False)
        '''

        if smoothen is None:
            smoothen = 0

        print("Initialising Dataset")

        self.path = path
        self.ids = selected_ids
        self.spacing = spacing
        self.image_size = image_size
        self.smoothen = smoothen

        # Save a couple of exemplar images from the dataset
        for img_idx in range(5):
            self.display(path, img_idx) # Save an example

        if edgen:  # resample_image(dts[0][0], [3, 3, 3], [60, 60, 50])
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.DiscreteGaussian(resample_image(sitk.ReadImage(PATH_TO_DATA + f"gm_volume3d/sub-{ID[0]}_ses-{ID[1]}_T2w_graymatter.nii.gz", sitk.sitkFloat32), self.spacing, self.image_size), smoothen)))).unsqueeze(0) for ID in self.ids]
        else:
            self.samples = [torch.from_numpy(sitk.GetArrayFromImage(sitk.DiscreteGaussian(resample_image(sitk.ReadImage(PATH_TO_DATA + f"gm_volume3d/sub-{ID[0]}_ses-{ID[1]}_T2w_graymatter.nii.gz", sitk.sitkFloat32), self.spacing, self.image_size), smoothen))).unsqueeze(0) for ID in self.ids]

        self.targets = torch.tensor(id_ages, dtype=torch.float).view((-1, 1))
        print("Initialisation complete")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.samples[item], self.targets[item]

    def display(self, path, item):
        from ..utils.utils import display_image
        ID = self.ids[item]
        img = sitk.SobelEdgeDetection(sitk.DiscreteGaussian(resample_image(sitk.ReadImage(PATH_TO_DATA + f"gm_volume3d/sub-{ID[0]}_ses-{ID[1]}_T2w_graymatter.nii.gz", sitk.sitkFloat32), self.spacing, self.image_size), self.smoothen))
        display_image(path, img, item)


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
    The main CNN
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