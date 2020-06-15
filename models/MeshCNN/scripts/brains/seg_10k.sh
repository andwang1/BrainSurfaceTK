#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/native_wm_10k_right \
--name seg_n_wm_10k_right \
--dataset_mode segmentation \
--arch meshunet \
--ninput_edges 30000 \
--ncf 8 16 \
--pool_res 2500 \
--batch_size 1 \
--lr 0.001 \

