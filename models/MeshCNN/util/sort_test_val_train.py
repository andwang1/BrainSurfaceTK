from tqdm import tqdm
from get_data_path import get_data_path
from shutil import copyfile
import sys
import pickle

def get_file_path(patient_id, session_id):
        file_name = "sub-" + patient_id +"_ses-" + session_id + files_ending
        file_path = data_folder + '/' + file_name
        return file_path


if __name__ == "__main__":

    path = sys.argv[1]

    with open('util/names.pk', 'rb') as f:
        indices = pickle.load(f)

    """
    {'Train': [patients...],
     'Val': [...],
    'Test': [...]  }
    """

    for patient_idx in tqdm(indices['Train']):
        file_name = patient_idx+".obj"
        copyfile(path+"obj/"+file_name, path+"train/"+file_name)

    for patient_idx in tqdm(indices['Val']):
        file_name = patient_idx+".obj"
        copyfile(path+"obj/"+file_name, path+"val/"+file_name)

    for patient_idx in tqdm(indices['Test']):
        file_name = patient_idx+".obj"
        copyfile(path+"obj/"+file_name, path+"test/"+file_name)


