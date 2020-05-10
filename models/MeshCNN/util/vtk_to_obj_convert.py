import os
import numpy as np
import pyvista as pv
from read_meta import read_meta
from get_edge_features import write_eseg, from_scratch, write_seseg, save_features

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

meta_data_path = "combined.tsv"
vtk_path = "datasets/reducedto_10k/pial/vtk/"
path = "datasets/seg_10k_left/"
obj_path = path + "obj/"
seg_path = path + "seg/"
sseg_path = path + "sseg/"
feat_path = path + "local_features/"

extension = "_left_pial_10k.vtk"
meta_data = read_meta(meta_data_path)
patient_names = []
ses_ids = []

for idx, ids in enumerate(meta_data):
    patient_id, ses_id = ids[0], ids[1]
    print(patient_id, ses_id)

    p = pv.Plotter(off_screen=True)

    try:
        meshv = pv.read(vtk_path + "sub-" + patient_id + "_ses-" + ses_id + extension)
    except:
        print("failed")
        continue
    p.add_mesh(meshv, show_edges=True)

    p.export_obj(obj_path + patient_id + "_" + ses_id)

    feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}
    vert_features = {feature: meshv.get_array(feature_arrays[feature]) for feature in feature_arrays.keys()}

    mesh_data = from_scratch(file=obj_path + patient_id + "_" + ses_id + ".obj", opt=None)

    write_eseg(mesh_data, vtk_path + "sub-" + patient_id + "_ses-" + ses_id + extension, seg_path, patient_id, ses_id)

    write_seseg(seg_path, sseg_path, patient_id, ses_id)

    save_features(mesh_data, vtk_path + "sub-" + patient_id + "_ses-" + ses_id + extension, feat_path, patient_id,
                  ses_id)
