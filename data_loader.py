import os
import os.path as osp

# import numpy as np
import pyvista as pv

from read_meta import read_meta
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_off
from torch_geometric.data import DataLoader
from torch_geometric.data import Data


class OurDataset(InMemoryDataset):
    def __init__(self, root, label_class='gender', classification=True, train=True, transform=None,
                        pre_transform=None, pre_filter=None, data_folder=None, add_birth_weight=True):

        self.train = train
        self.categories = {'gender': 2, 'birth_age': 3, 'weight': 4, 'scan_age': 6, 'scan_num': 7}
        self.meta_column_idx = self.categories[label_class]
        self.classes = dict()
        self.classification = classification
        self.add_birth_weight = add_birth_weight  # TODO: ADD OTHER FEATURES
        root += '/' + label_class

        # For reading .obj
        if data_folder is None:
            self.data_folder = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/'
        else:
            self.data_folder = data_folder

        super(OurDataset, self).__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        '''A list of files in the raw_dir which needs to be found in order to skip the download.'''
        return []

    @property
    def processed_file_names(self):
        '''A list of files in the processed_dir which needs to be found in order to skip the processing.'''
        return ['training.pt', 'test.pt']

    def download(self):
        '''No need to download data.'''
        pass

    def process(self):
        '''Processes raw data and saves it into the processed_dir.'''
        # Read data into huge `Data` list.
        if self.train:
            torch.save(self.process_set(), self.processed_paths[0])
        else:
            torch.save(self.process_set(), self.processed_paths[1])

    @staticmethod
    def get_file_path(patient_id, session_id, extension='vtp'):

        # repo = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-" \
        #        + patient_id + "/ses-" + session_id + "/anat/vtp"
        #
        # file_name = "sub-" + patient_id + "_ses-" + session_id \
        #         + "_hemi-L_space-dHCPavg32k_inflated_drawem_thickness_thickness_curvature_sulc_myelinmap_myelinmap."\
        #         + extension

        repo = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/inflated"
        file_name = "sub-"+ patient_id +"_ses-"+ session_id +"_hemi-L_inflated_reduce50.vtk"


        file_path = repo + '/' + file_name

        return file_path

    def process_set(self):

        # 0. Get meta data
        meta_data = read_meta()

        # 0. Get patient id number and label columns (0 = patient id, meta_column_idx = label column (eg. sex))
        if self.train:  # TODO: MAKE SPLIT CORRECTLY (ACCORDING TO INPUT IN PERCENTAGE). NOW TESTING == TRAINING
            meta_data = meta_data[80:, :]
        else:
            meta_data = meta_data[:80, :]

        # 1. Initialise the variables
        data_list = []
        categories = set(meta_data[:, self.meta_column_idx])           # Set of categories {male, female}

        # 2. Create category dictionary (mapping: category --> class), e.g. 'male' --> 0, 'female' --> 1
        for class_num, category in enumerate(categories):
            self.classes[category] = class_num

        # 3. Iterate through all patient ids
        for idx, patient_id in enumerate(meta_data[:, 0]):

            file_path = self.get_file_path(patient_id, meta_data[idx, 1])

            if os.path.isfile(file_path):

                mesh = pv.read(file_path)

                # Points
                points = torch.tensor(mesh.points)

                n_faces = mesh.n_cells
                faces = mesh.faces.reshape((n_faces, -1))
                faces = torch.tensor(faces[:, 1:].transpose())

                # Features # TODO: ADD ALL THE FEATURES THAT ARE NEEDED.
                x = None
                add_features = True
                if add_features:

                    corr_thickness = mesh.get_array(1)
                    curvature = mesh.get_array(3)
                    drawem = mesh.get_array(0)
                    sulc = mesh.get_array(4)
                    smoothed_myelin_map = mesh.get_array(2)

                    # print(corr_thickness.shape)
                    # print(curvature.shape)
                    # print(drawem.shape)
                    # print(sulc.shape)
                    # print(smoothed_myelin_map.shape)

                    # myelinMap = torch.tensor(mesh.get_array(6))
                    # array_2 = torch.tensor(mesh.get_array(2))

                    # Which features to add.
                    x = torch.tensor([corr_thickness, curvature, drawem, sulc, smoothed_myelin_map]).t()


                # classes[meta_data[:, 1][idx]] returns class_num from classes using key (e.g. 'female' -> 1)
                if self.classification:
                    y = torch.tensor([self.classes[meta_data[:, self.meta_column_idx][idx]]])
                else:
                    if self.add_birth_weight:
                        y = torch.tensor([[float(meta_data[:, self.meta_column_idx][idx]), float(meta_data[:, 4][idx])]])
                    else:
                        y = torch.tensor([[float(meta_data[:, self.meta_column_idx][idx])]])

                data = Data(x=x, pos=points, y=y, face=faces)
                data_list.append(data)
            else:
                continue

            #  KEEPING FOR NOW. Reading .obj files
            # Create path to file. _L_pial for now.
            # path = self.data_folder + patient_id + '_L_pial' +'.off'
            #
            # # Try read patient data
            # if os.path.isfile(path):
            #     data = read_off(path)
            #     data.y = torch.tensor([self.classes[meta_data[:, 2][idx]]])   # classes[meta_data[:, 1][idx]] returns class_num from classes using key (e.g. 'female' -> 1)
            #
            #     data_list.append(data)
            # else:
            #     continue

        # # 4. Attempt any pre-processing that is required
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)


if __name__ == '__main__':
    # Path to where the data will be saved.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    # Transformations, scaling and sampling 102 points (doesn't sample faces).
    pre_transform, transform = None, None  # T.NormalizeScale(), T.SamplePoints(1024) #T .FixedPoints(1024)

    myDataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform, label_class='scan_age', classification=False)

    print(myDataset)

    train_loader = DataLoader(myDataset, batch_size=1, shuffle=False)

    print(list(train_loader))

     # Printing dataset without sampling points. Will include faces.
    for i, (batch, face, pos, x, y) in enumerate(train_loader):
        # print(batch)
        # print(face[1].t())
        # print(pos)
        # print(x)
        print(y[1][0])
        print('_____________')
        break
