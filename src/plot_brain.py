import pickle
import os.path as osp

from src.pyvista_examples import plot

if __name__ == '__main__':

    path = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/deepl_brain_surfaces/experiment_data/aligned_sphere_50_-2'

    for brain_idx in range(1, 15):
        with open(path + f'/data_validation{brain_idx}.pkl', 'rb') as file:
            data, labels, pred = pickle.load(file)

            # print(len(data))
            plot(data, labels, pred)
#

