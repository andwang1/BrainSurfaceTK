import torch.utils.data as data
import numpy as np
import pickle
import os

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        super(BaseDataset, self).__init__()

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                              'ninput_channels': len(mean)}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']


#def collate_fn(batch):
#    """Creates mini-batch tensors
#    We should build custom collate_fn rather than using default collate_fn
#    """
#    # print("ENTER")
#    meta = {}
#    keys = batch[0].keys()
#    # print(batch[0]['edge_features'].shape)
#    # print(batch[1]['edge_features'].shape)
#    # print("BATCH", batch)
#    # print(batch[0]['edge_features'])
#    # print(batch[1]['edge_features'])
#    for key in keys:
#        if key=='edge_features':
#            arrays = []
#            for d in batch:
#                if d[key].shape[-1] != batch[0][key].shape[-1]:
#                    continue
#                arrays.append(d[key])
#            meta.update({key: np.array(arrays)})
#        else:
#            meta.update({key: np.array([d[key] for d in batch])})
#
#
#        #meta.update({key: np.array([d[key] for d in batch])})
#        #print('~~~~~~~~~~~~~~~~')
#        #print(len(meta))
#        #print(len(key))
#        #print([d[key] for d in batch])
#        #[print(type(d[key])) for d in batch]
# #       meta.update({key: np.array([d[key] for d in batch])})
#    # print("META", meta)
#    # print("LEAVE")
#    return meta

def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    # print("ENTER")
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        temp = [d[key] for d in batch]
        ## have to resize number of edge features when we have different shapes - fill with zeros
        if key == 'edge_features':
            max_num_faces = max([t.shape[1] for t in temp])
            [t.resize((5, max_num_faces), refcheck=False) for t in temp if t.shape[1] != max_num_faces]
            a = np.array(temp)
            meta.update({key: a})
        else:
            a = np.array(temp)
            meta.update({key: a})
    return meta

