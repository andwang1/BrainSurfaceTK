import os
import numpy as np
import pyvista as pv
from util.read_meta import read_meta
import sys

from util.get_edge_features import write_eseg, from_scratch, write_seseg, save_features
from util.normalize_labels import get_all_unique_labels

#pyvista docs: https://docs.pyvista.org/plotting/plotting.html#pyvista.BasePlotter.add_mesh

__author__ = "Francis Rhys Ward"
__license__ = "MIT"


if __name__ == '__main__':


    meta_data_path = sys.argv[1]
    vtk_path = sys.argv[2]
    path = sys.argv[3]
    extension = sys.argv[4]
    try:
        seg = sys.argv[5] == "seg"
    except:
        seg = False

    obj_path = path+"obj/"
    seg_path = path+"seg/"
    sseg_path = path+"sseg/"
    feat_path = path+"local_features/"

    dirs_to_create = [path, obj_path, seg_path, sseg_path, feat_path]
    for d in dirs_to_create:
        try:
            os.mkdir(d)
        except:
            print("directories already exist")

    meta_data = read_meta(meta_data_path)
    patient_names = []
    ses_ids = []

    label_mapping = get_all_unique_labels(meta_data, vtk_path, extension)

    print(label_mapping.values())
    print([0 if l is not 19 else 1 for l in label_mapping.values()])


    for idx, ids in enumerate(meta_data):
        patient_id, ses_id = ids[0], ids[1]
        print(patient_id, ses_id)

        p = pv.Plotter(off_screen=True)

        try:
            #print(vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension)
            meshv = pv.read(vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension)
        except:
            print("failed")
            continue
        p.add_mesh(meshv, show_edges=True)
        p.export_obj(obj_path+patient_id+"_"+ses_id)
        if seg:

            with open(path+'classes.txt', 'w') as f:
                for label in label_mapping.values():
                f.write("%s\n" % label)

            feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}
            vert_features = {feature:meshv.get_array(feature_arrays[feature]) for feature in feature_arrays.keys()}

            mesh_data = from_scratch(file=obj_path+ patient_id+"_"+ses_id+".obj", opt=None)

            write_eseg(label_mapping, mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+  ses_id+extension, seg_path, patient_id, ses_id)
            write_seseg(label_mapping, seg_path, sseg_path, patient_id, ses_id)
            save_features(mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension, feat_path, patient_id, ses_id)
