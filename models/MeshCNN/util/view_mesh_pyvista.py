import pyvista as pv
import sys

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

if __name__ == "__main__":
    file_path = sys.argv[1]
    print("*********************")
    print(file_path)
    mesh = pv.read(file_path)
    surf_faces = pv.PolyData(mesh.points, mesh.faces)
    plotter = pv.Plotter()
    surf_points = pv.PolyData(mesh.points[[4185, 4181, 4179]])
    plotter.add_mesh(surf_points, show_edges=True, render_points_as_spheres=True, point_size=10, color='red')
    plotter.add_mesh(surf_faces)

    plotter.show()

