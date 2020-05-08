<<<<<<< README.md
# DeepL_Brain_Surfaces
# MeshCNN

MeshCNN is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. 
This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

Github repo can be found here: https://github.com/ranahanocka/MeshCNN/

We have made multiple modifcations, including functionality for regression, adding global features into the penultimate fully-connected layers, adding logging of test-ouput, and functionality for new learning-rate schedulers. 

# GUI

The main server has been developed using the Django framework. Please note that the folder **main** actually contains all the main project code & static files. **BasicSite** acts as the head of the server. 

Have set up mod_wsgi to serve static files.
See documentation at: [https://pypi.org/project/mod-wsgi/](https://pypi.org/project/mod-wsgi/)

To set up the modwsgi, please run: 
`python -m pip install mod_wsgi-standalone`

To run the server, use: 
`python manage.py runmodwsgi --url-alias /media media`

Package-wise:

*  Nibabel is used to load the raw MRI patient files
*  Nilearn is used to output an interactive slicing tool for the loaded MRI patient files.
*  VTK.js is used to load & interactively display .vtp brain surface files.


TODO:
*  It appears that we have a package conflict between PyVista & Mod-WSGI. This is problematic since both are crucial.
*  Check that in production, the upload feature works