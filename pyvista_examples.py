import pyvista as pv


patient_id = "CC00069XX12"
repo = "/vol/project/2019/545/g1954504/Vitalis/data/vtk_files/"
file_name = "sub-" + patient_id + "_hemi-L_space-dHCPavg32k_inflated_drawem_thickness_thickness_curvature_sulc_myelinmap_myelinmap.vtk"

file_path = repo + file_name


mesh = pv.read(file_path)

n_faces = mesh.n_cells

print(mesh)
print(mesh.points)
print(mesh.n_arrays)
print(mesh.get_array(0))
faces = mesh.faces.reshape((n_faces, -1))
faces = faces[:, 1:]
print(faces)

surf_faces = pv.PolyData(mesh.points, mesh.faces)
surf_points = pv.PolyData(mesh.points)


surf_points['values'] = mesh.get_array(3)
surf_faces['values'] = mesh.get_array(4)

surf_points.plot(eye_dome_lighting=True, render_points_as_spheres=True)
surf_faces.plot(eye_dome_lighting=True, show_edges=False)
surf_faces.plot(eye_dome_lighting=True, show_edges=True)
