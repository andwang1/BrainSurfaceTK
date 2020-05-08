import open3d as o3d
import numpy as np
import copy

mesh = o3d.io.read_triangle_mesh("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/alex/data/ModelNet10/raw/bathtub/train/bathtub_0001.off")
print(mesh)
print(np.asarray(mesh.vertices))
print(np.asarray(mesh.triangles))
print("")

print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])
print("A mesh with no normals and no colors does not seem good.")

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

print("We make a partial mesh of only the first half triangles.")
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) //
                                        2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])

print("Painting the mesh")
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1])