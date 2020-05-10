import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
from data.get_feature_dict import get_feature_dict

__author__ = "Rana Hanocka"
__license__ = "MIT"
__maintainer__ = ["Andy Wang", "Francis Rhys Ward"]

class ClassificationData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        if opt.verbose:
            print("DEBUG dir path in dataset ", self.dir)
        if opt.dataset_mode == 'regression':
            self.class_to_idx = get_feature_dict(opt.label)
            self.nclasses = 1
        else:
            self.classes, self.class_to_idx = self.find_classes(self.dir)
            self.nclasses = len(self.classes)

        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase, opt.dataset_mode)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label, 'path': path}
        # get edge features
        edge_features = mesh.extract_features()
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset_by_class(self, dir, class_to_idx, phase, dataset_mode):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.opt.verbose:
                        print("DEBUG fname in meshretrieve ", fname)
                    if is_mesh_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        if self.opt.verbose:
                            print("DEBUG meshretrieve path ", path)
                        if dataset_mode == 'regression':
                            # Retrieves additional info from metadata file as labels - use filename as key
                            # filename format CC00839XX23_23710.obj
                            filename_key = fname[:-4]
                            # for the specific case where we are using both halves in the same dataset the file will end in L or R
                            if filename_key[-1] in ('L', 'R'):
                                filename_key = filename_key[:-2]
                            item = (path, class_to_idx[filename_key])
                        else:
                            item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes
