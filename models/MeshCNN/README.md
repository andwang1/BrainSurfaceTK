# MeshCNN
The original MeshCNN GitHub repositary can be found here: https://github.com/ranahanocka/MeshCNN/ in which the MeshCNN architecture is implemented in PyTorch. 

We recommend viewing the original README before using this repo as we will describe here only how to use the added functioanlity that we have implemented.

# Regression

We implemented regression due to interest in predicting age from brain-surface meshes. Regression is a global task, i.e. akin to classification (with one label per mesh), and as opposed to segmentation which is implemented as an edge-wise classification task. 

To access this functionality simply set the `--dataset_mode` to regression in the train/test scripts, and set the label to be a feature from the `util/meta_data.tsv` file. 

# Global Features

We similarly add in global features to the network by specifying the `--features` option in the scripts to match a column of the `meta_data.tsv` file. 

# Learning Rate Schedulers

We also added functionality for different learning rate schedulers, this can be accessed by setting `--lr_policy` in training scripts, to the standard `lambda|step|plateau` options we added `cyclic|cosine_restarts|static`. 

# Upweighting

Furthermore, for binary classification tasks (e.g. classification of preterm births) we added functionality to upweight the minority class with the     `--weight_minority` option. 

# Viewing the Mesh with Pyvista

Lastly, we use the PyVista package to view the mesh, including collapsed meshes produced by MeshCNN and key vertices. 