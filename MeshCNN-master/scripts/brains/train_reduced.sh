#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reduced_90_gender \
--name brains \
--epoch_count 1 \
--niter 5 \
--batch_size 14 \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--niter_decay 100 \
--verbose_plot \
--dataset_mode regression \
--print_freq 14 \

