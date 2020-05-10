import pyvista as pv

file_name = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/deepl_brain_surfaces/rand/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtk"
file_path = file_name
mesh = pv.read(file_path)

n_faces = mesh.n_cells

faces = mesh.faces.reshape((n_faces, -1))
faces = faces[:, 1:]

surf_faces = pv.PolyData(mesh.points, mesh.faces)
surf_points = pv.PolyData(mesh.points)
drawem = mesh.get_array(0).copy()

surf_faces['values'] = drawem

# surf_faces.save('new_file.vtp')

new_faces = pv.PolyData('forCem.vtp')

new_faces.plot()