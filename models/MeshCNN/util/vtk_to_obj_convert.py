import os
import numpy as np
import pyvista as pv
from read_meta import read_meta

from get_edge_features import write_eseg, from_scratch, write_seseg, save_features

#pyvista docs: https://docs.pyvista.org/plotting/plotting.html#pyvista.BasePlotter.add_mesh

if __name__ == '__main__':


    import sys
    meta_data_path = sys.argv[1]
    vtk_path = sys.argv[2]
    path = sys.argv[3]
    try:
        seg = sys.argv[4] == "seg"
    except:
        seg = False
    #meta_data_path = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/combined.tsv"
    #vtk_path = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_10k/pial/vtk/"
    #path = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/deepl_brain_surfaces/MeshCNN-master/datasets/seg_10k_left/"
    obj_path = path+"obj/"
    seg_path = path+"seg/"
    sseg_path = path+"sseg/"
    feat_path = path+"local_features/"

    extension = "_left_pial_10k.vtk"
    meta_data = read_meta(meta_data_path)
    patient_names = []
    ses_ids = []


    for idx, ids in enumerate(meta_data):
        patient_id, ses_id = ids[0], ids[1]
        print(patient_id, ses_id)

        p = pv.Plotter(off_screen=True)

        try:
            meshv = pv.read(vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension)
        except:
            print("failed")
            continue
        p.add_mesh(meshv, show_edges=True)

        p.export_obj(obj_path+patient_id+"_"+ses_id)

        if seg:
            feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}
            vert_features = {feature:meshv.get_array(feature_arrays[feature]) for feature in feature_arrays.keys()}

            mesh_data = from_scratch(file=obj_path+ patient_id+"_"+ses_id+".obj", opt=None)

            write_eseg(mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+  ses_id+extension, seg_path, patient_id, ses_id)

            write_seseg(seg_path, sseg_path, patient_id, ses_id)

            save_features(mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension, feat_path, patient_id, ses_id)
