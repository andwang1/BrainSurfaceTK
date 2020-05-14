#!/usr/bin/env bash

pushd models/MeshCNN

## run the test and export collapses
python3 test.py \
--dataroot datasets/brains \
--checkpoints_dir checkpoints/reg \
--export_folder checkpoints/mesh_collapses \
--name brains \
--label scan_age \
--ncf 16 32 \
--pool_res 3000 2750 \
--norm group \
--num_groups 2 \
--dataset_mode regression \
--verbose_plot \
--which_epoch 1 \

popd

