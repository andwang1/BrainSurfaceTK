import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh
import pickle
import re

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)

        if opt.dataset_mode == 'regression':
            self.load_patient_age_dict()
        else:
            # ORIG CODE
            self.classes, self.class_to_idx = self.find_classes(self.dir)

        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase, opt.dataset_mode, self.retrieve_patient_and_session)
        # ORIG CODE
        if opt.dataset_mode == 'regression':
            self.nclasses = 1
        else:
            self.nclasses = len(self.classes)

        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        # print("DEBUG path", path)
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label, 'path': path}
        # get edge features
        edge_features = mesh.extract_features()
        #edge_features = pad(edge_features, self.opt.ninput_edges)
        # print("DEBUG shape", edge_features.shape)
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

    # removed staticmethod
    def make_dataset_by_class(self, dir, class_to_idx, phase, dataset_mode, retrieve_patient_func):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        if dataset_mode == 'regression':
                            # needs filename starting with CCblabla_SESSID
                            item = (path, class_to_idx[retrieve_patient_func(fname)])
                        else:
                            item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes

    def retrieve_patient_and_session(self, fname):
        re_pattern = "(\w+)_(\d+)\.obj"
        result = re.search(re_pattern, fname)
        patient_name = result.group(1)
        session_name = result.group(2)
        return patient_name + "_" + session_name

    def load_patient_age_dict(self):
        with open(r"/vol/biomedic2/aa16914/shared/MScAI_brain_surface/andy/patient_to_age.pk", "rb") as f:
            self.class_to_idx = pickle.load(f)
