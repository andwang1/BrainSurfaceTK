import pickle
import os
import pandas as pd
from shutil import copyfile

# Where them brains at
source_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/all_brains"
# Where them brains should be at
target_dir = r"/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/datasets/brains_reg_red90"

#### This is for MeshCNN specifically
try:
    os.chdir(target_dir)
except FileNotFoundError:
    os.makedirs(f"{target_dir}/Male/train")
    os.makedirs(f"{target_dir}/Male/test")
    os.makedirs(f"{target_dir}/Female/train")
    os.makedirs(f"{target_dir}/Femle/test")
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
train_indices = indices["Train"] + indices["Val"]
test_indices = indices["Test"]

for patient in train_indices:
    file_name = f"{patient}.obj"
    gender = meta.loc[patient]['gender']
    dest_path = f"{target_dir}/{gender}/train"
    copyfile(f"{source_dir}/{file_name}", dest_path)

for patient in test_indices:
    file_name = f"{patient}.obj"
    gender = meta.loc[patient]['gender']
    dest_path = f"{target_dir}/{gender}/test"
    copyfile(f"{source_dir}/{file_name}", dest_path)
