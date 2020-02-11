#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/brains_reduced0 \
--name brains \
--batch_size 1 \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
