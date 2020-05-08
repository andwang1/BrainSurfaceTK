import os
import os.path as osp

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from read_meta import read_meta
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_off
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T


class OurDataset(InMemoryDataset):
    def __init__(self, root, label_class='gender', classification=True, train=True, transform=None,
                        pre_transform=None, pre_filter=None, data_folder=None):

        self.train = train
        self.categories = {'gender': 2, 'birth_age': 3, 'weight': 4, 'scan_age': 6, 'scan_num': 7}
        self.meta_column_idx = self.categories[label_class]
        self.classes = dict()
        self.classification = classification
        root += '/' + label_class

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
        '''A list of files in the prcessed_dir which needs to be found in order to skip the processing.'''
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

    def process_set(self):

        # 0. Get meta data
        meta_data = read_meta()

        patient_names = []

        # 0. Get patient id number and label columns (0 = patient id, meta_column_idx = label column (eg. sex))
        if self.train:  # TODO: MAKE SPLIT CORRECTLY (ACCORDING TO INPUT IN PERCENTAGE). NOW TESTING == TRAINING
            meta_data = meta_data[:, [0, self.meta_column_idx]]
        else:
            meta_data = meta_data[:, [0, self.meta_column_idx]]

        # 1. Initialise the variables
        data_list = []
        categories = set(meta_data[:, 1])           # Set of categories {male, female}

        # 2. Create category dictionary (mapping: category --> class), e.g. 'male' --> 0, 'female' --> 1
        for class_num, category in enumerate(categories):
            self.classes[category] = class_num

        # 3. Iterate through all patient ids
        for idx, patient_id in enumerate(meta_data[:, 0]):
            # TODO: DECIDE WHERE TO PUT ALL THE DATA, NAMING CONVENTIONS, WHICH DATA TO USE.
            repo = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/vtk_files/"
            file_name = "sub-" + patient_id + "_hemi-L_space-dHCPavg32k_inflated_drawem_thickness_thickness_curvature_sulc_myelinmap_myelinmap_reduce90.vtk"
            file_path = repo + file_name

            if os.path.isfile(file_path):
                patient_names.append(patient_id)
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(file_path)
                reader.Update()

                # Points
                points = torch.tensor(np.array(reader.GetOutput().GetPoints().GetData()))

                # Getting the faces  # TODO: THERE ARE SOME DIFFERENCES. DISCUSS WITH AMIR
                cells = reader.GetOutput().GetPolys()
                nCells = cells.GetNumberOfCells()
                array = cells.GetData()
                nCols = array.GetNumberOfValues() // nCells
                numpy_cells = vtk_to_numpy(array)
                numpy_cells = numpy_cells.reshape((-1, nCols))
                faces = torch.tensor(numpy_cells[:, [1, 2, 3]].transpose())

                # Features # TODO: ADD ALL THE FEATURES THAT ARE NEEDED.
                x = None
                add_features = True
                if add_features:
                    corr_thickness = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(1))
                    curvature = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(3))
                    drawem = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(0))
                    sulc = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(4))
                    smoothed_myelin_map = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(5))
                    myelinMap = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(6))
                    # array_2 = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(2))  # TODO: ASK ABOUT THIS

                    # Which features to add.
                    x = torch.tensor([corr_thickness, curvature, drawem, sulc, smoothed_myelin_map, myelinMap]).t()


                # classes[meta_data[:, 1][idx]] returns class_num from classes using key (e.g. 'female' -> 1)
                if self.classification:
                    y = torch.tensor([self.classes[meta_data[:, 1][idx]]])
                else:
                    y = torch.tensor([float(meta_data[:, 1][idx])])

                data = Data(x=x, pos=points, y=y, face=faces)
                data_list.append(data)
            else:
                continue

            # #  KEEPING FOR NOW
            # #Create path to file. _L_pial for now.
            # path = self.data_folder + patient_id + '_L_pial' +'.off'
            #
            # # Try read patient data
            # if os.path.isfile(path):
            #     data = read_off(path)
            #     data.y = torch.tensor([self.classes[meta_data[:, 1][idx]]])   # classes[meta_data[:, 1][idx]] returns class_num from classes using key (e.g. 'female' -> 1)
            #
            #     data_list.append(data)
            # else:
            #     continue

        # # 4. Attempt any pre-processing that is required
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # print('@@@@@@@ ', data_list)
        # for i in data_list:
        #     print(type(i))
        return self.collate(data_list)
        # print(data_list)
        # return data_list


if __name__ == '__main__':
    # Path to where the data will be saved.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    # Transformations, scaling and sampling 102 points (doesn't sample faces).
    pre_transform, transform = None, None  # T.NormalizeScale(), T.SamplePoints(1024) #T .FixedPoints(1024)

    myDataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform)

    print(myDataset)

    train_loader = DataLoader(myDataset, batch_size=1, shuffle=False)

    print(list(train_loader))

     # Printing dataset without sampling points. Will include faces.
    for i, (batch, face, pos, x, y) in enumerate(train_loader):
        print(batch)
        print(face)
        print(pos)
        print(x)
        print(y)
        print('_____________')
        break
