#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_red50 \
--checkpoints_dir checkpoints/red50_12gb_batch2 \
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
--batch_size 2 \
--ncf 64 112 128 \
--pool_res 3000 2500 2000 \
