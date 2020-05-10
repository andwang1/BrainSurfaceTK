# MeshCNN
The original MeshCNN GitHub repositary can be found here: https://github.com/ranahanocka/MeshCNN/ in which the MeshCNN architecture is implemented in PyTorch. 

We recommend viewing the original README before using this repo as we will describe here only how to use the added functionality that we have implemented.

# .vtk to .obj Conversion

Firstly, we have python scripts to convert a directory of files from `.vtk` to `.obj` (as required by MeshCNN), whilst also creating the `.seseg` and `.eseg` files needed for segmentation. This module takes sys args `meta_data_path` (a path to the `.tsv` file containing meta data), `vtk_path` to where the .vtk files are stored, `path` to where the .obj files should be stored, and optionally `seg` to also save the files necessary for segmentation. 

# Regression

We implemented regression due to interest in predicting age from brain-surface meshes. Regression is a global task, i.e. akin to classification (with one label per mesh), and as opposed to segmentation which is implemented as an edge-wise classification task. 

To access this functionality simply set the `--dataset_mode` to regression in the train/test scripts, and set the `--label` to be a feature from the `util/meta_data.tsv` file. 

# Global Features

Similarly global features can be inputted into the final fully-connected layers of the network by specifying the `--features` option in the scripts to match a column-heading of the `meta_data.tsv` file. 

# Learning Rate Schedulers

We also added functionality for different learning rate schedulers, this can be accessed by setting `--lr_policy` in training scripts; to the standard `lambda|step|plateau` options we added `cyclic|cosine_restarts|static`. 


# Viewing the Mesh with Pyvista

Lastly, we use the PyVista package to view the mesh, including collapsed meshes produced by MeshCNN and key vertices. 

# Acknowledgements 

For the MeshCNN paper see  Rana Hanocka et al. “MeshCNN: A Network with an Edge” (2018). url:https://arxiv.org/abs/1809.05910.