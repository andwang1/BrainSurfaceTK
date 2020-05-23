#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/seg_test \
--name brain_seg_test \
--dataset_mode segmentation \
--arch meshunet \
--ninput_edges 30000 \
--ncf 2 2 \
--pool_res 2000 \
--batch_size 1 \
--lr 0.001 \
