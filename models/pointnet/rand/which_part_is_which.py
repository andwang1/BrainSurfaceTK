import pyvista as pv
import numpy as np



#### THIS SCRIPT IS A NICE WAY TO VISUALISE DIFFERENT PARTS OF THE BRAIN




# Default values
mode = None

print('='*50)
print('Welcome to Which-Is-Which!')

while mode is None:
    mode = input('Would you like to visualise the brain using points or meshes? [p/m] (default=meshes)')
    if mode == 'p':
        pass
    elif mode == 'm':
        pass
    elif mode == '':
        mode = 'm'
    else:
        mode = None
        print('Select either [p / m / Enter].')


file_name = "./random/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtk"
file_path = file_name
mesh = pv.read(file_path)

n_faces = mesh.n_cells

faces = mesh.faces.reshape((n_faces, -1))
faces = faces[:, 1:]

surf_faces = pv.PolyData(mesh.points, mesh.faces)
surf_points = pv.PolyData(mesh.points)
drawem = mesh.get_array(0).copy()

not_over = True

while not_over:

    label_mode = input('You can select ALL labels to view or a particular one. Simply type "all" or the number you want (0-17): ')

    if label_mode == 'all':
        for label in np.unique(drawem):

            for idx, value in enumerate(drawem):
                if value != label:
                    drawem[idx] = 100

            surf_points['values'] = drawem
            surf_faces['values'] = drawem

            if mode == 'p':
                surf_points.plot(eye_dome_lighting=False, render_points_as_spheres=True)
            elif mode == 'm':
                surf_faces.plot(eye_dome_lighting=False, show_edges=False)

            # surf_faces.plot(eye_dome_lighting=True, show_edges=True)

            drawem = mesh.get_array(0).copy()

    else:
        label_mode = np.unique(drawem)[int(label_mode)]
        for idx, value in enumerate(drawem):
            if value != label_mode:
                drawem[idx] = 100

        surf_points['values'] = drawem
        surf_faces['values'] = drawem

        if mode == 'p':
            surf_points.plot(eye_dome_lighting=False, render_points_as_spheres=True)
        elif mode == 'm':
            surf_faces.plot(eye_dome_lighting=False, show_edges=False)

        drawem = mesh.get_array(0).copy()


        x = input('Do you want to go again? [y/n]')

    if x == 'y':
        pass
    elif x == 'n':
        not_over = False