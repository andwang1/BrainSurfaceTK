import pickle
import os
import pandas as pd
from shutil import copyfile

__author__ = “Andy Wang”
__license__ = “MIT”

# Where the brains at
source_dir = r"datasets/all_brains_merged_10k"
# Where the brains should be at
target_dir = r"datasets/brains_cls_binary_preterm_merged_10k"

#### This is for MeshCNN specifically
if not os.access(target_dir, mode=os.F_OK):
    os.makedirs(f"{target_dir}/preterm/train")
    os.makedirs(f"{target_dir}/preterm/test")
    os.makedirs(f"{target_dir}/preterm/val")
    os.makedirs(f"{target_dir}/not_preterm/train")
    os.makedirs(f"{target_dir}/not_preterm/test")
    os.makedirs(f"{target_dir}/not_preterm/val")
####


# Load indices
with open("names_preterm_04152020_noCrashSubs.pk", "rb") as f:
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
        dest_path = f"{target_dir}/preterm/val/{file_name}"
    else:
        dest_path = f"{target_dir}/not_preterm/val/{file_name}"
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
        dest_path = f"{target_dir}/preterm/test/{file_name}"
    else:
        dest_path = f"{target_dir}/not_preterm/test/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")
