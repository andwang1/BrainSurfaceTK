import pyvista as pv
import numpy as np
import pickle
path = '/Users/brandelt/Dropbox/EDUCATION/Imperial College/MScAI/010 Software Project/main_repository/deepl_brain_surfaces/experiment_data/aligned_inflated_50_-2'


with open(path + f'/data_validation1.pkl', 'rb') as file:
    data, labels, pred = pickle.load(file)

    data = data[:data.shape[0] // 2]
    labels = labels[:labels.shape[0] // 2]
    pred = pred[:pred.shape[0] // 2]

# Create and structured surface
mesh = pv.PolyData(data)
mesh_target = pv.PolyData(data)


# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 1)
plotter.add_text("Target", font_size=30)
plotter.add_mesh(mesh_target, scalars=labels, render_points_as_spheres=True)

plotter.subplot(0, 0)
plotter.add_text("Predictions", font_size=30)
plotter.add_mesh(mesh, scalars=pred, render_points_as_spheres=True)

print('Orient the view, then press "q" to close window and produce movie')

# setup camera and close
plotter.show(auto_close=False)
plotter.get_default_cam_pos(negative=True)
path_orbit = plotter.generate_orbital_path(n_points=36, shift=mesh.length//2)

# Open a gif
plotter.open_gif("wave3.gif")
pts = mesh.points.copy()

for phase in range(1, 150):

    with open(path + f'/data_validation{phase}.pkl', 'rb') as file:
        data, labels, pred = pickle.load(file)
        pred = pred[:pred.shape[0] // 2]

    # plotter.update_coordinates(pts)
    plotter.subplot(0, 0)
    plotter.update_scalars(pred)
    plotter.write_frame()

    if phase % 10 == 0:
        plotter.orbit_on_path(path_orbit)

# Close movie and delete object
plotter.close()