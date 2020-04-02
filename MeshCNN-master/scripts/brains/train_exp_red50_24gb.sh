#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_red50 \
--checkpoints_dir checkpoints/red50_24gb \
--export_folder checkpoints/mesh_collapses \
--name brains \
--ninput_edges 48735 \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode regression \
--niter 5 \
--niter_decay 100 \
--batch_size 1 \
--ncf 1024 1024 2048 2048 3072  \
--pool_res 40000 30000 20000 10000 5000 \
