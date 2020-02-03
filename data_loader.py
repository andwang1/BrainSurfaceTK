import os
import os.path as osp
import shutil
import glob
from deepl_brain_surfaces.read_meta import read_meta
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_off

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        # self.pt_path = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/processed_off/'

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

        print('Processed paths: ', self.processed_paths)

        self.categories = ['gender', 'scan_age']
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        '''A list of files in the raw_dir which needs to be found in order to skip the download.'''
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        '''A list of files in the processed_dir which needs to be found in order to skip the processing.'''
        return ['training.pt', 'test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        '''Processes raw data and saves it into the processed_dir.'''
        # Read data into huge `Data` list.
        print('PROCESSING')
        print(self.processed_paths[0])
        torch.save(self.process_set('train'), self.processed_paths[0])
        # torch.save(self.process_set('test'), '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/processed_off')


    def process_set(self, dataset):

        # 0. Get column number of meta data corresponding to 'sex' (or other categories)
        # meta_column_idx = self.categories.index('gender')
        meta_column_idx = 2

        # 0. Get meta data
        meta_data = read_meta()

        # 0. Get patient id number and label columns (0 = patient id, meta_column_idx = label column (eg. sex))
        meta_data = meta_data[:, [0, meta_column_idx]]

        # 1. Initialise the variables
        folder = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/'
        data_list = []
        categories = set(meta_data[:, 1])           # Set of categories {male, female}

        # 2. Create category dictionary (mapping: category --> class), e.g. 'male' --> 0, 'female' --> 1
        classes = dict()
        for class_num, category in enumerate(categories):
            classes[category] = class_num

        # 3. Iterate through all patient ids
        for idx, patient_id in enumerate(meta_data[:, 0]):

            # Create path to file
            path = folder + patient_id + '_L_pial' +'.off'

            # Try read patient data
            if os.path.isfile(path):
                # print('Reading patient: {}'.format(patient_id))
                data = read_off(path)
                data.y = torch.tensor([classes[meta_data[:, 1][idx]]])   # classes[meta_data[:, 1][idx]] returns class_num from classes using key (e.g. 'female' -> 1)

                # print(classes[meta_data[:, 1][idx]])
                # print('='*30)
                # print('='*30)
                # print('Patient ', patient_id)
                # print(data)
                # print(data.y)

                data_list.append(data)
            else:
                continue


        # # 4. Attempt any pre-processing that is required
        # if self.pre_filter is not None:
        #     data_list = [d for d in data_list if self.pre_filter(d)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(d) for d in data_list]


        data, slices = self.collate(data_list)
        print(data, '\n\n\n', slices)

        return self.collate(data_list)

        # categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        # print('Raw directory is ', self.raw_dir)
        # categories = sorted([x.split(os.sep)[-2] for x in categories])
        # print('Our categories are ', categories)
        #
        # data_list = []
        # for target, category in enumerate(categories):
        #     folder = osp.join(self.raw_dir, category, dataset)                  # raw directory + 'bathtub' + 'train' or 'test'
        #     paths = glob.glob('{}/{}_*.off'.format(folder, category))           # PATHS = a list of paths to files /bathtub_0001.off, etc.
        #     for path in paths:
        #         data = read_off(path)                                           # data.pos contains all the POINTS (point cloud)
        #         data.y = torch.tensor([target])                                 # data.y contains all the LABELS
        #         data_list.append(data)
        #
        # if self.pre_filter is not None:
        #     data_list = [d for d in data_list if self.pre_filter(d)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(d) for d in data_list]
        #
        # return self.collate(data_list)

if __name__ == '__main__':
    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')

    myDataset = MyOwnDataset(path)
    myDataset.process()