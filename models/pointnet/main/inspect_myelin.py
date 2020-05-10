import pickle
import os
import os.path as osp
from tqdm import tqdm
import pyvista as pv
from ..src.utils import get_data_path
PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'


data_folder, files_ending = get_data_path(data_nativeness='native', data_compression='30k', data_type='white', hemisphere='left')

def get_file_path(patient_id, session_id):
    file_name = "sub-" + patient_id + "_ses-" + session_id + files_ending
    file_path = data_folder + '/' + file_name
    return file_path


with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
    indices = pickle.load(f)


myelinless_patients = []
for mode in ['Train', 'Val', 'Test']:
    print(f'Checking all arrays in {mode}')
    for patient_idx in tqdm(indices[mode]):
        for array in ['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']:

            file_path = get_file_path(patient_idx[:11], patient_idx[12:])

            # If file exists
            if os.path.isfile(file_path):
                mesh = pv.read(file_path)

                # IF array is empty
                if mesh.get_array('myelin_map') is None:
                    myelinless_patients.append(f'{patient_idx}')

print('Patients without myelin: {}'.format(myelinless_patients))