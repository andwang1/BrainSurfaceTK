#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reduced_90_gender \
--checkpoints_dir checkpoints/classification_models \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--batch_size 1 \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--verbose_plot \
--dataset_mode classification \
