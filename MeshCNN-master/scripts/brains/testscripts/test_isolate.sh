#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/isolate_error \
--checkpoints_dir checkpoints/test \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 1 \
--seed 0 \
--dataset_mode regression \
--niter 100000 \
--niter_decay 0 \
--batch_size 1 \
--ncf 64 64 64 \
--pool_res 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \
--verbose \
