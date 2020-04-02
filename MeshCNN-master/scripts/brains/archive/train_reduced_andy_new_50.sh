#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_red50 \
--checkpoints_dir checkpoints/red50_run2 \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--niter 5 \
--batch_size 1 \
--ncf 64 112 128 \
--pool_res 3000 2500 2000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--verbose_plot \
--dataset_mode regression \
--print_freq 14 \
--ninput_edges 9000 \
