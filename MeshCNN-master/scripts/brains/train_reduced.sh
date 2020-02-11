#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/brains_reduced_90_gender \
--name brains \
--epoch_count 5 \
--niter 20 \
--batch_size 1 \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 1 \
--niter_decay 100 \
--verbose_plot \
