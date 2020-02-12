#!/usr/bin/env bash

python3 train.py \
--dataroot datasets/brains_m_f \
--name brains \
--ninput_edges 64980 \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--batch_size 1 \
#--help
