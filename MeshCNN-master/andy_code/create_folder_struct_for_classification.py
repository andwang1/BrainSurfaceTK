import pickle
import os
import pandas as pd
from shutil import copyfile
# Where them brains at
source_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/all_brains_50"
# Where them brains should be at
target_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/brains_cls_binary_preterm_red50f"
target_test_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/brains_cls_binary_preterm_red50_testf"

#### This is for MeshCNN specifically
if not os.access(target_dir, mode=os.F_OK):
    os.makedirs(f"{target_dir}/preterm/train")
    os.makedirs(f"{target_dir}/preterm/test")
    os.makedirs(f"{target_dir}/not_preterm/train")
    os.makedirs(f"{target_dir}/not_preterm/test")
    os.makedirs(f"{target_test_dir}/preterm/train")
    os.makedirs(f"{target_test_dir}/not_preterm/train")
####


# Load indices
with open("names_preterm.pk", "rb") as f:
    indices = pickle.load(f)

# Load metadata
meta = pd.read_csv("combined.tsv", delimiter='\t')

# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)

train_indices = indices["Train"]
val_indices = indices["Val"]
test_indices = indices["Test"]

file_counter = 0

preterm_age = 38

for patient in train_indices:
    file_name = f"{patient}.obj"
    source_path = f"{source_dir}/{file_name}"
    birth_age = meta.loc[patient]['birth_age']
    is_preterm = birth_age <= preterm_age
    if is_preterm:
        dest_path = f"{target_dir}/preterm/train/{file_name}"
    else:
        dest_path = f"{target_dir}/not_preterm/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")

for patient in val_indices:
    file_name = f"{patient}.obj"
    source_path = f"{source_dir}/{file_name}"
    birth_age = meta.loc[patient]['birth_age']
    is_preterm = birth_age <= preterm_age
    if is_preterm:
        dest_path = f"{target_dir}/preterm/test/{file_name}"
    else:
        dest_path = f"{target_dir}/not_preterm/test/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")

for patient in test_indices:
    file_name = f"{patient}.obj"
    source_path = f"{source_dir}/{file_name}"
    birth_age = meta.loc[patient]['birth_age']
    is_preterm = birth_age <= preterm_age
    if is_preterm:
        dest_path = f"{target_test_dir}/preterm/train/{file_name}"
    else:
        dest_path = f"{target_test_dir}/not_preterm/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")
