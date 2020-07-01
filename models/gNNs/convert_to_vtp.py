import os
from concurrent.futures import ThreadPoolExecutor

import pyvista as pv
from tqdm import tqdm

"""
TODO: Keep features as well :)
"""

def convert_to_vtp(path, new_path, num_workers=0):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    else:
        answer = input("This new_path already exists. Do you want to overwrite it's contents? (y/n)")
        if answer == "y":
            old_files = [os.path.join(new_path, fp) for fp in os.listdir(new_path)]
            for old_file in old_files:
                if os.path.isfile(old_file) and old_file.endswith(".vtp"):
                    os.unlink(old_file)
        else:
            raise IsADirectoryError("Aborting! new_path already exists, please choose a different new_path.")
    fps = [os.path.join(path, fp) for fp in os.listdir(path)]
    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as e:
            for fp in tqdm(fps):
                e.submit(convert_file_to_vtp, fp, new_path)
    else:
        for fp in tqdm(fps):
            convert_file_to_vtp(fp, new_path)


def convert_file_to_vtp(filepath, new_path):
    mesh = pv.read(filepath)
    surf_mesh = pv.PolyData(mesh.points, mesh.faces)
    drawem = mesh.get_array(0).copy()
    surf_mesh['values'] = drawem
    surf_mesh.save(os.path.join(new_path, os.path.basename(filepath)).replace(".vtk", ".vtp"))


if __name__ == "__main__":
    path = "/mnt/UHDD/Programming/Projects/DeepLearningOnBrains/models/gNNs/tmp_data/vtks"
    # path = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native/reduced_20k/white"
    new_path = "/mnt/UHDD/Programming/Projects/DeepLearningOnBrains/models/gNNs/tmp_data/vtps"
    # new_path = "/vol/bitbucket/cnw119/data/vtps"
    convert_to_vtp(path, new_path, 0)
