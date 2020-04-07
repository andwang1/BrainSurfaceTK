import pickle
import os.path as osp

from src.pyvista_examples import plot

if __name__ == '__main__':

    # path = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/deepl_brain_surfaces/experiment_data/aligned_sphere_50_-2'
    path = '/Users/brandelt/Dropbox/EDUCATION/Imperial College/MScAI/010 Software Project/main_repository/deepl_brain_surfaces/experiment_data/aligned_sphere_50_-2'

    for brain_idx in range(1, 35):
        try:
            with open(path + f'/data_validation{brain_idx}.pkl', 'rb') as file:
                data, labels, pred = pickle.load(file)
                print(data.shape)
                data = data[:data.shape[0]//2]
                labels = labels[:labels.shape[0]//2]
                pred = pred[:pred.shape[0]//2]


                # print(len(data))
                plot(data, labels, pred)
        except:
            pass

