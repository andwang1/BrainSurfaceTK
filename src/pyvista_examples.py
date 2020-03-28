import pyvista as pv
import numpy as np

#
# patient_id = "CC00069XX12"
# repo = "/vol/project/2019/545/g1954504/Vitalis/data/vtk_files/"
# file_name = "sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtk"
#
# file_path = file_name
#
#
# mesh = pv.read(file_path)
#
# n_faces = mesh.n_cells
#
# # print(mesh)
# # print(mesh.points)
# # print(mesh.n_arrays)
# # print(mesh.get_array(0))
# faces = mesh.faces.reshape((n_faces, -1))
# faces = faces[:, 1:]
# # print(faces)
#
# surf_faces = pv.PolyData(mesh.points, mesh.faces)
# surf_points = pv.PolyData(mesh.points)
# drawem = mesh.get_array(0).copy()
#
# for label in np.unique(drawem):
#
#     for idx, value in enumerate(drawem):
#         if value != label:
#             drawem[idx] = 100
#
#     surf_points['values'] = drawem
#     # surf_faces['values'] = mesh.get_array(0)
#
#     surf_points.plot(eye_dome_lighting=False, render_points_as_spheres=True)
#     # surf_faces.plot(eye_dome_lighting=True, show_edges=False)
#     # surf_faces.plot(eye_dome_lighting=True, show_edges=True)
#
#     drawem = mesh.get_array(0).copy()

def plot(points, labels, predictions):
    print(type(points), type(labels), type(predictions))
    surf_points = pv.PolyData(points)
    surf_points['values'] = labels
    surf_points.plot(eye_dome_lighting=False, render_points_as_spheres=True)

    surf_points['values'] = predictions
    surf_points.plot(eye_dome_lighting=False, render_points_as_spheres=True)

# def

