#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/seg_test \
--name brain_seg_test \
--arch meshunet \
--dataset_mode segmentation \
--ninput_edges 30000 \
--ncf 32 64 128 256 \
--pool_res 4000 3000 2000 \
--resblocks 3 \
--batch_size 1 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
