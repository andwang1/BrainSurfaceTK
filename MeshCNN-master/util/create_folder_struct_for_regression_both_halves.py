import pickle
import os
import pandas as pd
from shutil import copyfile
# Where them brains at
source_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/all_brains_50_LR"
# Where them brains should be at
target_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/brains_reg_red50_LR"
target_test_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/brains_reg_red50_LR_test"

#### This is for MeshCNN specifically
if not os.access(target_dir, mode=os.F_OK):
    os.makedirs(f"{target_dir}/Male/train")
    os.makedirs(f"{target_dir}/Male/test")
    os.makedirs(f"{target_dir}/Female/train")
    os.makedirs(f"{target_dir}/Female/test")
    os.makedirs(f"{target_test_dir}/Male/train")
    os.makedirs(f"{target_test_dir}/Female/train")
####


# Load indices
with open("indices.pk", "rb") as f:
    indices = pickle.load(f)

# Load metadata
meta = pd.read_csv("combined.tsv", delimiter='\t')

# Put participant_id and session_id together to get a unique key and use as index
meta['unique_key'] = meta['participant_id'] + "_" + meta['session_id'].astype(str)
meta.set_index('unique_key', inplace=True)

# This will merge train and val
train_indices = indices["Train"]
val_indices = indices["Val"]
test_indices = indices["Test"]

file_counter = 0

for patient in train_indices:
    file_name = f"{patient}_L.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_dir}/{gender}/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")
    file_name = f"{patient}_R.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_dir}/{gender}/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")

for patient in val_indices:
    file_name = f"{patient}_L.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_dir}/{gender}/test/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")
    file_name = f"{patient}_R.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_dir}/{gender}/test/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")

for patient in test_indices:
    file_name = f"{patient}_L.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_test_dir}/{gender}/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")
    file_name = f"{patient}_R.obj"
    gender = meta.loc[patient]['gender']
    source_path = f"{source_dir}/{file_name}"
    dest_path = f"{target_test_dir}/{gender}/train/{file_name}"
    print("Attempting copy source", source_path)
    print("Attempting copy dest", dest_path)
    copyfile(source_path, dest_path)
    file_counter += 1
    print(f"Copy success, file {file_counter}")