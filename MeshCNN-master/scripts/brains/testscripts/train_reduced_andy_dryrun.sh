#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_dryrun \
--checkpoints_dir checkpoints/dryrun \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--niter 1 \
--niter_decay 20 \
--batch_size 1 \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--norm group \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--verbose_plot \
--dataset_mode regression \
--print_freq 14 \
--ninput_edges 10000 \
--min_lr 5e-5 \
--verbose \
