from nilearn.surface import load_surf_mesh
import os

def load_obj_from_vtp(patient,surface_type="pial", hemisphere="L"):

    os.chdir(f"/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-{patient}/")
    session = os.listdir(path='../../andy_code')[0]
    # print(session)
    session_number = session[4:]

    surf_file = f'/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-{patient}/ses-{session_number}/anat/sub-{patient}_ses-{session_number}_hemi-{hemisphere}_space-dHCPavg32k_{surface_type}.surf.gii'

    surf_mesh = load_surf_mesh(surf_file)

    text = ""
    for row in surf_mesh[0]:
        row_item = "v"
        for index, coordinate in enumerate(row):
            row_item += f" {coordinate}"
        row_item += '\n'
        text += row_item

    text += '\n'

    for row in surf_mesh[1]:
        row_item = "f"
        for index, coordinate in enumerate(row):
            row_item += f" {coordinate}"
        row_item += '\n'
        text += row_item

    os.chdir("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meshcnn_data")
    with open(f"{patient}_{hemisphere}_{surface_type}.obj", "w") as f:
        f.write(text)

patient_names = ["CC00099AN18", "CC00117XX10", "CC00122XX07", "CC00126XX11", "CC00138XX15",  "CC00162XX06", "CC00164XX08", "CC00168XX12", "CC00170XX06"]
# patient_names = [ "CC00126XX11"]

for patient in patient_names:
    load_obj_from_vtp(patient)
