from data_loader_v2 import OurDataset
import os.path as osp
import os
from read_meta import read_meta


if __name__ == '__main__':
    # root to where the data will be saved.
    path = "/vol/biomedic/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_90/vtk/pial"
    root = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/datasets/vtk_to_obj/new_data/tensors_90"

    myDataset = OurDataset(root, train=True)

    print(myDataset)

    meta_data = read_meta()
    patient_names = []
    ses_ids = []

    for idx, ids in enumerate(meta_data):
        patient_id, ses_id = ids[0], ids[1]
        # TODO: DECIDE WHERE TO PUT ALL THE DATA, NAMING CONVENTIONS, WHICH DATA TO USE.
        repo = "/vol/biomedic/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_90/vtk/pial"
        file_name = "sub-" + patient_id + "_" + ses_id + "_hemi-L_pial_reduce90.vtk"
        file_path = repo + file_name

        patient_names.append(patient_id)
        ses_ids.append(ses_id)



    for idx, data in enumerate(myDataset):
        text = ""
        for row in data.pos:
            row_item = "v"
            for index, coordinate in enumerate(row):
                row_item += f" {coordinate}"
            row_item += '\n'
            text += row_item

        text += '\n'
        for row in data.face.t():

            row_item = "f"
            for index, coordinate in enumerate(row):
                row_item += f" {coordinate}"
            row_item += '\n'
            text += row_item

        os.chdir("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/datasets/vtk_to_obj/90")
        with open(f"{patient_names[idx]}_{ses_ids[idx]}.obj", "w") as f:
            f.write(text)
