# Deep Learning on Brain Surfaces

# Setting up
To install all required packages, please setup a virtual environment as per the instructions below. This virtual environment is based on a CUDA 10.1.105 installation.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements1.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements2.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```

Alternatively, for a CPU installation, please setup the virtual environment as per the instructions below. Please note that the MeshCNN model requires the CUDA based installation above.
```
python3 -m venv venv
source venv/bin/activate
pip install -r cpu_requirements1.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -r cpu_requirements2.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html
``` 


# MeshCNN

MeshCNN is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. 
This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

The original GitHub repo and additional run instructions can be found here: https://github.com/ranahanocka/MeshCNN/

In this repository, we have made multiple modifcations. These include functionality for regression, adding global features into the penultimate fully-connected layers, adding logging of test-ouput, allowing for a train/test/validation split, and functionality for new learning-rate schedulers among other features.

###### Run instructions

Place the .obj mesh data files into a folder in *models/MeshCNN/datasets* with the correct folder structure - below is an example of the structure. Here, *brains* denotes the name of the directory in *models/MeshCNN/datasets* which holds one directory for each class, here e.g. *Male* and *Female*.
In each class, folders *train*, *val* and *test* hold the files.

<img src="https://gitlab.doc.ic.ac.uk/aw1912/deepl_brain_surfaces/-/raw/master/img/meshcnn_data.png" width="450" height="263" />

<!--![MeshCNN dir struct](<src>)-->


<!--Place the data into the data set folder with the correct folder structure. Below is an example of the structure. *brains* denotes the name of the directory -->
<!--![MeshCNN dir struct](https://gitlab.doc.ic.ac.uk/aw1912/deepl_brain_surfaces/-/raw/master/img/meshcnn_data.png)-->


From the main repository level, the model can then be trained using, e.g. for regression
```
./scripts/regression/MeshCNN/regression.sh
```



# PointNet

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