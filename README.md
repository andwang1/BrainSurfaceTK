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


# GUI

This server has been developed using the [Django](https://www.djangoproject.com/) framework. We use [MOD-WSGI](https://modwsgi.readthedocs.io/en/develop/) Standalone to run an Apache server to host this webapp.

The MRI visualisation is done thanks to [Nilearn](https://nilearn.github.io/index.html) and the Brain surface is displayed using [VTK.js](https://kitware.github.io/vtk-js/index.html).

###### Run instructions

After following the instructions on creating a virtual environment containing all of our dependencies:
1. First you will want to create a super user. This can be done by: 
```
chmod 700 ./createsuperuser.sh
./createsuperuser.sh
```
2. Next you may want to use your own original data, this can be done by overwriting the meta_data.tsv data file in ``GUI/media/original/data``, please take care that the column names are exactly the same and in the same order. If not, then the load data function that can be called in the admin panel will not work. 
3. After you've created a super user, you can either run the server in developement mode by running:
```
python startserver.py
```
4. Alternatively you may wish to run the server in production mode. If you want others to remotely access this server, you may need to open port 8000 on your machine and please don't forget to port forward if you are using a modem. After you have done this, you can simply run:
```
python startserver.py prod
```

# PointNet++
PointNet++ is a hierarchical neural network, which was proposed to be used on point-cloud geometric data [1] for the tasks of regression, classification, and segmentation. In this project, we apply this architecture onto point-cloud representations of brain surfaces to tackle the tasks of age regression and brain segmentation.

###### Run instructions

The run instructions differ slightly for Pointnet regression and segmentation. Please proceed to the README in models/pointnet of this repository for full information.


# MeshCNN

MeshCNN [2] is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. 
This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

The original GitHub repo and additional run instructions can be found here: https://github.com/ranahanocka/MeshCNN/

In this repository, we have made multiple modifcations. These include functionality for regression, adding global features into the penultimate fully-connected layers, adding logging of test-ouput, allowing for a train/test/validation split, and functionality for new learning-rate schedulers among other features.

###### Run instructions

Place the .obj mesh data files into a folder in *models/MeshCNN/datasets* with the correct folder structure - below is an example of the structure. Here, *brains* denotes the name of the directory in *models/MeshCNN/datasets* which holds one directory for each class, here e.g. *Male* and *Female*.
In each class, folders *train*, *val* and *test* hold the files.

<img src="https://gitlab.doc.ic.ac.uk/aw1912/neodeepbrain/-/raw/master/img/meshcnn_data.png" width="450" height="263" />

Please additionally place a file called *meta_data.tsv* in the *models/MeshCNN/util* folder. This tab-seperated file will be used to read in additional labels and features into the model.
The file should contain columns participant_id and session_id, which will be concatenated to form a unique identifier of a patient's scan. This unique identifier must be used to name the data files in the datasets/ folder structure described above.
E.g. a *meta_data.tsv* file might look like this:

participant_id	session_id	scan_age

CC00549XX22	100100	42.142347

The corresponding mesh data file must then be named
*CC00549XX22_100100.obj*

Any continuous-valued columns in the *meta_data.tsv* file can then be used as features or labels in the regression using switches in the training file, as mentioned below.
```
--label scan_age
--features birth_age
```

From the main repository level, the model can then be trained using, e.g. for regression
```
./scripts/regression/MeshCNN/train_reg_brains.sh
```
Similarly, a pretrained model can be applied to the test set, e.g.
```
./scripts/regression/MeshCNN/test_reg_brains.sh
```

###### References
[1] Qi, C.R., Yi, L., Su, H., & Guibas, L.J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NIPS.
[2] Rana Hanocka et al. (2019). MeshCNN: A Network with an Edge. SIGGRAPH 2019.