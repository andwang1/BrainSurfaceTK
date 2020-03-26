import os
import pandas as pd
from shutil import copyfile

os.chdir("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/datasets/vtk_to_obj/90")

obj_files = os.listdir(".")

labels = pd.read_csv("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv", delimiter='\t')

for file_name in obj_files:
    if file_name in ["Male", "Female"]:
        continue
    patient_name = file_name[:11]

    gender = labels["gender"][labels["participant_id"] == patient_name]
    gender = list(gender)[0]
    print(gender)
    copyfile(f"/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/datasets/vtk_to_obj/90/{file_name}", f"/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/datasets/brains_reduced_90_gender/{gender}/{file_name}")
