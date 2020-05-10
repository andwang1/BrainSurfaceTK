import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import pickle

#####
# This script checks the duplicates within and between each of the split folds
#####

path_to_names = PATH_TO_ROOT + '/pointnet/src/names.pk'

with open(path_to_names, 'rb') as file:
    names = pickle.load(file)

unique_patients = [[], [], []]
duplicate_patients = []
number_of_occurences = []

# Check duplicates within the splits
for mode_idx, mode in enumerate(names):
    for patient in names[mode]:

        patient = patient.split('_')
        patient_id, session_id = patient

        if patient_id in unique_patients[mode_idx]:
            print(f'Duplicate within {mode}: {patient_id}')
        else:
            unique_patients[mode_idx].append(patient_id)

# Check duplicates between splits
for mode_idx, mode in enumerate(names):
    for mode_idx2, mode2 in enumerate(names):

        if mode_idx == mode_idx2:
            continue

        for patient in names[mode]:
            patient = patient.split('_')
            patient_id, session_id = patient

            for patient2 in names[mode2]:
                patient2 = patient2.split('_')
                patient_id2, session_id2 = patient2

                if patient_id == patient_id2:
                    print(f'Duplicate between {mode} and {mode2}: {patient_id}_{session_id} amd {patient_id2}_{session_id2}')
                else:
                    unique_patients[mode_idx].append(patient_id)
