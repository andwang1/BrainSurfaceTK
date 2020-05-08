#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/all_brains_find_error_merged_10k \
--checkpoints_dir checkpoints/test \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 1000 \
--seed 0 \
--dataset_mode regression \
--niter 1 \
--niter_decay 0 \
--batch_size 1 \
--ncf 2  \
--pool_res 3000 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \
