#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/all_brains_without_error_merged_10k \
--export_folder checkpoints/mesh_collapses \
--checkpoints_dir checkpoints/testnew \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 2 \
--dataset_mode regression \
--niter 10 \
--niter_decay 0 \
--batch_size 1 \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \
--verbose \
