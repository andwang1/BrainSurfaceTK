#!/usr/bin/env bash

## run the test and export collapses
python3 test.py \
--dataroot datasets/brains_testing \
--checkpoints_dir checkpoints/testing \
--export_folder checkpoints/mesh_collapses \
--name brains \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--norm group \
--export_folder meshes \
--dataset_mode regression \
--verbose_plot \
--seed 0 \
--which_epoch 5 \
