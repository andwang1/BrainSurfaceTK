#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_full \
--checkpoints_dir checkpoints/fullres \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--niter 5 \
--batch_size 1 \
--ncf 32 112 128 \
--pool_res 2000 1500 1000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--verbose_plot \
--dataset_mode regression \
--print_freq 10 \
--ninput_edges 9000 \
