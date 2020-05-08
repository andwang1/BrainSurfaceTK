from utils.utils import read_meta
from utils.utils import clean_data, get_ids_and_ages
import matplotlib.pyplot as plt


# 1. What are you predicting?
categories = {'gender': 3, 'birth_age': 4, 'weight': 5, 'scan_age': 7, 'scan_num': 8}
meta_column_idx = categories['scan_age']

# 2. Read the data and clean it
meta_data = read_meta()
meta_data = clean_data(meta_data)

# 3. Get a list of ids and ages (labels)
unique_patients = []
duplicate_patients = []
number_of_occurences = []
for idx, patient_id in enumerate(meta_data[:, 1]):

    if patient_id in duplicate_patients:
        continue

    if patient_id in unique_patients:
        pass
    else:
        unique_patients.append(patient_id)

    count = list(meta_data[:, 1]).count(patient_id)

    if count > 1:

        duplicate_patients.append(patient_id)
        number_of_occurences.append(count)


print(f'Total number of unique patients: {len(unique_patients)}')
print(f'Total number of duplicate patients: {len(duplicate_patients)}')

for patient, num in zip(duplicate_patients, number_of_occurences):

    print(f'\t Patient {patient}: {num}')