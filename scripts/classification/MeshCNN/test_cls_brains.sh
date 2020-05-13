#!/usr/bin/env bash

pushd models/MeshCNN

## run the test and export collapses
python3 test.py \
--dataroot datasets/brains \
--checkpoints_dir checkpoints/cls \
--export_folder checkpoints/mesh_collapses \
--name brains \
--features scan_age \
--ncf 16 32 \
--pool_res 3000 2750 \
--norm group \
--dataset_mode binary_class \
--verbose_plot \
--which_epoch 1 \

popd
