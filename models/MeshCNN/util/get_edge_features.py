import pickle
import pyvista as pv
import sys
from models.layers import mesh_prepare

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

sys.path.insert(1, 'models/layers')
feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}


def get_vert_features(vtk_path):
    """
    param: vtk_path
    returns a dictionary with keys as feature string and values as a list of feature values for each vertex
    """
    mesh = pv.read(vtk_path)
    feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}
    vert_features = {feature: mesh.get_array(feature_arrays[feature]) for feature in feature_arrays.keys()}
    return vert_features


def get_edge_features(mesh_data, feature_list, vtk_path):
    """
    mesh_data is a meshcnn object used to create the mesh, see mesh_prepare.py
    feature_list is the list of local features you want to extract e.g. "drawem", "curvature"
    the function creates an attribute for the mesh_data object called edge_local_features which is a dict from features to a list of values in the same order as mesh_data.edges
    returns this dict
    """
    vert_features = get_vert_features(vtk_path)
    mesh_data.edge_local_features = {feature: [] for feature in feature_arrays.keys()}
    for edge in mesh_data.edges:
        for feature in feature_arrays.keys():
            if feature != "drawem":
                vertex_feature_vals = vert_features[feature]
                avg_vert_feature = (vertex_feature_vals[edge[0]] + vertex_feature_vals[edge[1]]) / 2
                mesh_data.edge_local_features[feature].append(avg_vert_feature)
        mesh_data.edge_local_features["drawem"].append(
            vert_features["drawem"][edge[0]])  # just take one label for each edge
    return mesh_data.edge_local_features


def write_eseg(mesh_data, vtk_path, seg_path, patient_id, ses_id):
    get_edge_features(mesh_data, ["drawem"], vtk_path)
    edge_seg_labels = mesh_data.edge_local_features["drawem"]

    eseg_path = seg_path + patient_id + "_" + ses_id + ".eseg"
    with open(eseg_path, 'w') as f:
        for label in edge_seg_labels:
            f.write("%s\n" % label)


def write_seseg(eseg_path, seseg_path, patient_id, ses_id):
    eseg_file = eseg_path + patient_id + "_" + ses_id + ".eseg"
    seseg_file = seseg_path + patient_id + "_" + ses_id + ".seseg"
    labels = range(38)

    with open(eseg_file) as f:
        eseg = f.read().splitlines()
    with open(seseg_file, 'w') as f:
        for label in eseg:
            row = [0 if l is not int(label) else 1 for l in labels]
            f.write(str(row).strip("[]").replace(",", ""))
            f.write("\n")


def save_features(mesh_data, vtk_path, feat_path, patient_id, ses_id):
    feature_names = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    features = get_edge_features(mesh_data, feature_names, vtk_path)

    save_file = feat_path + patient_id + "_" + ses_id + "_local_features.p"
    pickle.dump(features, open(save_file, "wb"))
