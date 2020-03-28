import os
import os.path as osp

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset

from src.read_meta import read_meta


class OurDataset(InMemoryDataset):
    def __init__(self, root, task='classification', target_class='gender', train=True, transform=None,
                        pre_transform=None, pre_filter=None, data_folder=None, add_features=True, local_features=[],
                        global_feature=[], files_ending=None, test_size=0.1, random_state=42, reprocess=False,
                        val_size=-1.0, val=False):
        '''
        Creates a Pytorch dataset from the .vtk/.vtp brain data.

        :param root: Root path, where processed data objects will be placed
        :param task: Possible tasks include: {'classification', 'regression', 'segmentation'}
        :param target_class: Target class that is being predicted (if applicable).
        :param train: If true, returns train data
        :param transform: Transformation applied
        :param pre_transform: Pre-transformation applied
        :param pre_filter: Pre-filter applied
        :param data_folder: Path to the data folder with the dataset
        :param add_features: If true, adds all features from .vtp/.vtk files to x in Dataset
        :param test_size: relative size of the training set
        :param random_state: Random seed for recreating results
        :param reprocess: Flag to reprocess the data even if it was processed before and saved in the root folder.
        :param local_features: Local features that should be added to every point.
        :param global_feature: Global features that should be added to the label for later use.
        :param val_size: size of the validation set (fraction of the remaining training data after train/test split).
               IF THIS IS NOT ZERO, THE PROCESSING IS DONE FOR VALIDATION SET.
        '''

        # Train, test, validation
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.val_size = val_size
        self.val = val

        # Metadata categories
        self.categories = {'gender': 2, 'birth_age': 3, 'weight': 4, 'scan_age': 6, 'scan_num': 7}
        self.meta_column_idx = self.categories[target_class]

        # Classes dict. Populated later. Saved in case you need to look this up.
        self.classes = dict()

        # Mapping between features and array number in the files.
        self.feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}

        # The task at hand
        self.task = task

        # Additional global features
        self.local_features = local_features
        self.global_feature = global_feature

        # Other useful variables
        self.add_features = add_features
        self.unique_labels = []
        self.num_labels = 0
        self.reprocess = reprocess

        # Initialise path to data
        if data_folder is None:
            self.data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/inflated"
        else:
            self.data_folder = data_folder

        if files_ending is None:
            self.files_ending = "_hemi-L_inflated_reduce50.vtk"
        else:
            self.files_ending = files_ending

        super(OurDataset, self).__init__(root, transform, pre_transform, pre_filter)

        # Standard paths to processed data objects (train or test or val)

        if self.train:
            path = self.processed_paths[0]
        elif self.val:
            path = self.processed_paths[2]
        else:
            path = self.processed_paths[1]

        # If processed_paths exist, return without having to process again
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        '''A list of files in the raw_dir which needs to be found in order to skip the download.'''
        return []


    @property
    def processed_file_names(self):
        '''A list of files in the processed_dir which needs to be found in order to skip the processing.
        if self.reprocess, doesn't skip processing'''

        if self.reprocess:
            return ['training.pt', 'test.pt', 'validation.pt', 'a']

        return ['training.pt', 'test.pt', 'validation.pt']


    def download(self):
        '''No need to download data.'''
        pass


    def process(self):
        '''Processes raw data and saves it into the processed_dir.'''
        # Read data into huge `Data` list.

        if self.train:
            torch.save(self.process_set(), self.processed_paths[0])
        else:
            if self.val:
                torch.save(self.process_set(), self.processed_paths[2])
            else:
                torch.save(self.process_set(), self.processed_paths[1])


    def get_file_path(self, patient_id, session_id):

        file_name = "sub-" + patient_id +"_ses-" + session_id + self.files_ending
        file_path = self.data_folder + '/' + file_name

        return file_path


    def get_features(self, list_features, mesh):
        '''Returns tensor of features to add in every point.
        :param list_features: list of features to add. Mapping is in self.feature_arrays
        :param mesh: pyvista mesh from which to get the arrays.
        :returns: tensor of features or None if list is empty.'''
        # TODO: ASK AMIR TO SORT THAT DRAWEM CLASS IMBALANCE OUT

        # Very ugly workaround about some classes not being in some data.
        list_of_drawem_labels = [0, 5, 7, 9, 11, 13, 15, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39]

        if list_features:

            if 'drawem' in list_features:
                one_hot_drawem = pd.get_dummies(mesh.get_array(self.feature_arrays['drawem']))

                new_df = pd.DataFrame()
                for label in list_of_drawem_labels:
                    if label not in one_hot_drawem.columns:
                        new_df[label] = 0
                    else:
                        new_df[label] = one_hot_drawem[label]

                one_hot_drawem = new_df.to_numpy()

                drawem_list = [one_hot_drawem[:, i] for i in range(one_hot_drawem.shape[1])]

            else:
                drawem_list = []

            features = [mesh.get_array(self.feature_arrays[key]) for key in self.feature_arrays if key != 'drawem']

            return torch.tensor(features + drawem_list).t()
        else:
            return None


    def get_global_features(self, list_features, meta_data, patient_idx):
        '''Returns list of global features to add to label, later to be used in fully connected layers.
        :param list_features: list of features to add. Mapping is in self.categories
        :param meta_data: meta_data.
        :param patient_idx: index of the patient from the metadata.
        :return list: list of features from meta data.'''
        return [float(meta_data[patient_idx, self.categories[feature]]) for feature in list_features]


    def split_data(self, meta_data):
        '''Split data into training and testing maintaining the same distribution for labels. Splitting is done
        based on the task and the target labels.
        :param meta_data: meta_data that is going to be split.
        :returns: Train or test meta_data.'''

        if self.test_size == 0:
            if self.train:
                return meta_data
            else:
                raise ValueError('Test size is zero. Can not create test data')

        if self.task == 'regression':
            _, bins = np.histogram(meta_data[:, self.meta_column_idx].astype(float), bins='doane')
            y_binned = np.digitize(meta_data[:, self.meta_column_idx].astype(float), bins)

            X_train, X_test, y_train, y_test = train_test_split(meta_data, meta_data[:, self.meta_column_idx],
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                stratify=y_binned)
            if self.val_size > 0:
                _, bins = np.histogram(X_train[:, self.meta_column_idx].astype(float), bins='doane')
                y_binned = np.digitize(X_train[:, self.meta_column_idx].astype(float), bins)

                X_train, X_val, y_train, y_test = train_test_split(X_train, X_train[:, self.meta_column_idx],
                                                                    test_size=self.val_size,
                                                                    random_state=self.random_state,
                                                                    stratify=y_binned)

        else:
            X_train, X_test, y_train, y_test = train_test_split(meta_data, meta_data[:, self.meta_column_idx],
                                                                test_size=self.test_size,
                                                                random_state=self.random_state,
                                                                stratify=meta_data[:, self.meta_column_idx])
            if self.val_size > 0:
                X_train, X_val, y_train, y_test = train_test_split(X_train, X_train[:, self.meta_column_idx],
                                                                   test_size=self.val_size,
                                                                   random_state=self.random_state,
                                                                   stratify=X_train[:, self.meta_column_idx])
        if self.train:
            return X_train
        else:
            if self.val:
                return X_val
            else:
                return X_test


    def clean_data(self, meta_data):
        '''Cleans the meta_data. Removes rows in the data that we don't have files from.
        :param meta_data: meta data.
        :return data: cleaned version of meta data'''

        missing_idx = []

        for idx, patient_id in enumerate(meta_data[:, 0]):
            file_path = self.get_file_path(patient_id, meta_data[idx, 1])
            if not os.path.isfile(file_path):
                missing_idx.append(idx)

        # for idx in missing_idx:
        meta_data = np.delete(meta_data, missing_idx, 0)

        return meta_data


    def get_all_unique_labels(self, meta_data):
        '''
        Return unique mapping of drawem features such that
            Original: [0, 3, 5, 7, 9 ...]
            Required: [0, 1, 2, 3, 4 ...]
            Mapping: [0:0, 3:1, 5:2, 7:3, 9:4, ...]
        :return: Mapping
        '''
        ys = []
        lens = []
        # 3. Iterate through all patient ids
        for idx, patient_id in enumerate(meta_data[:, 0]):

            # Get file path to .vtk/.vtp for one patient
            file_path = self.get_file_path(patient_id, meta_data[idx, 1])

            # If file exists
            if os.path.isfile(file_path):

                mesh = pv.read(file_path)
                y = torch.tensor(mesh.get_array(0))
                lens.append(y.size(0))
                ys.append(y)

        # Now process the uniqueness of ys
        ys_concatenated = torch.cat(ys)
        unique_labels = torch.unique(ys_concatenated)
        unique_labels_normalised = unique_labels.unique(return_inverse=True)[1]

        # Create the mapping
        label_mapping = {}
        for original, normalised in zip(unique_labels, unique_labels_normalised):
            label_mapping[original.item()] = normalised.item()

        return label_mapping


    def normalise_labels(self, y_tensor, label_mapping):
        '''
        Normalises labels in the format necessary for segmentation
        :return: tensor vector of normalised labels ([0, 3, 1, 2, 4, ...])
        '''
        # Having received y_tensor, use label_mapping
        temporary_list = []
        for y in y_tensor:
            temporary_list.append(label_mapping[y.item()])
        return torch.tensor(temporary_list)


    def process_set(self):
        '''Reads and processes the data. Collates the processed data which is later saved.'''
        # 0. Get meta data
        meta_data = read_meta()

        # Cleaning data. Dropping rows, that we don't have files for.
        meta_data = self.clean_data(meta_data)

        # Get the mapping for the entire dataset, in order to normalise DRAWEM labels for segmentation
        label_mapping = self.get_all_unique_labels(meta_data)

        # Splitting data.
        meta_data = self.split_data(meta_data)

        # 1. Initialise the variables
        data_list = []
        categories = set(meta_data[:, self.meta_column_idx])           # Set of categories {male, female}

        # 2. Create category dictionary (mapping: category --> class), e.g. 'male' --> 0, 'female' --> 1
        for class_num, category in enumerate(categories):
            self.classes[category] = class_num

        # 3. These lists will collect all the information for each patient in order
        lens = []
        xs = []
        poss = []
        ys = []
        faces_list = []

        # 3. Iterate through all patient ids
        for idx, patient_id in enumerate(meta_data[:, 0]):

            # Get file path to .vtk/.vtp for one patient
            file_path = self.get_file_path(patient_id, meta_data[idx, 1])

            # If file exists
            if os.path.isfile(file_path):

                mesh = pv.read(file_path)

                # Get points
                points = torch.tensor(mesh.points)

                # Get faces
                n_faces = mesh.n_cells
                faces = mesh.faces.reshape((n_faces, -1))
                faces = torch.tensor(faces[:, 1:].transpose())

                # Features
                x = self.get_features(self.local_features, mesh)

                # Global features
                global_x = self.get_global_features(self.global_feature, meta_data, idx)

                # Generating label based on the task. By default regression.
                if self.task == 'classification':
                    y = torch.tensor([[self.classes[meta_data[idx, self.meta_column_idx]]] + global_x])

                elif self.task == 'segmentation':
                    y = torch.tensor(mesh.get_array(0))

                    # Retrieve the lengths of each label tensor (i.e. number of labelled points)
                    lens.append(y.size(0))

                # Else, regression
                else:
                    y = torch.tensor([[float(meta_data[idx, self.meta_column_idx])] + global_x])

                # Add the data to the lists
                xs.append(x)
                poss.append(points)
                ys.append(y)
                faces_list.append(faces)

        # Now process the uniqueness of ys
        ys_normalised = self.normalise_labels(torch.cat(ys), label_mapping)
        ys = ys_normalised.split(lens)

        # Now add all patient data to the list
        for x, points, y, faces in zip(xs, poss, ys, faces_list):
            # Create a data object and add to data_list
            data = Data(x=x, pos=points, y=y, face=faces)
            data_list.append(data)

        # Do any pre-processing that is required
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # # Keeping information to look up later.
        # if self.task == 'segmentation':
        #
        #     y = torch.cat(ys).unique(return_inverse=True)[1]
        #
        #     # Get a set of unique labels (already standardized)
        #     self.unique_labels = torch.cat(self.unique_labels).unique()
        #
        #     # Get the number of unique labels
        #     self.num_labels = len(self.unique_labels)

        return self.collate(data_list)


if __name__ == '__main__':

    # Path to where the data will be saved.
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/test_reduce')

    data_folder = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_90/vtk/inflated"
    files_ending = "_hemi-L_inflated_reduce90.vtk"
    # Transformations, scaling and sampling 102 points (doesn't sample faces).
    pre_transform, transform = None, None  # T.NormalizeScale(), T.SamplePoints(1024) #T .FixedPoints(1024)

    # myDataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
    #                        target_class='scan_age', task='regression', reprocess=True,
    #                        local_features=['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']
    #                        , global_feature=['weight'], data_folder=data_folder, files_ending=files_ending,
    #                        val_size=0.1, test_size=0.09)
    #
    # myDataset2 = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform,
    #                        target_class='scan_age', task='regression', reprocess=True,
    #                        local_features=['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']
    #                        , global_feature=['weight'], data_folder=data_folder, files_ending=files_ending,
    #                        val_size=0.1, test_size=0.09)

    myDataset3 = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
                            target_class='scan_age', task='regression', reprocess=False,
                            local_features=['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']
                            , global_feature=['weight'], data_folder=data_folder, files_ending=files_ending,
                            val_size=0.1, test_size=0.09, val=True)

    # print(myDataset)
    # print(myDataset2)
    print(myDataset3)
    print(myDataset3[0].x.size(1))
    print(myDataset3[0].y.size(1))

    # train_loader = DataLoader(myDataset, batch_size=1, shuffle=False)
    # train_loader2 = DataLoader(myDataset2, batch_size=1, shuffle=False)
    train_loader3 = DataLoader(myDataset3, batch_size=1, shuffle=False)

    #
    #  # Printing dataset without sampling points. Will include faces.
    # for i, (batch, face, pos, x, y) in enumerate(train_loader3):
    #     # print(batch)
    #     print(face[1].size())
    #     # print(pos[1].size())
    #     # print(x)
    #     # print(y)
    #     print('_____________')
