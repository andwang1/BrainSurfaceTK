import pyvista as pv
import sys

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

file_path = sys.argv[1]
mesh = pv.read(file_path)
surf_faces = pv.PolyData(mesh.points, mesh.faces)
plotter = pv.Plotter()
surf_points = pv.PolyData(mesh.points[[4185, 4181, 4179]])
plotter.add_mesh(surf_points, show_edges=True, render_points_as_spheres=True, point_size=10, color='red')
plotter.add_mesh(surf_faces)

if __name__ == "__main__":
    plotter.show()

