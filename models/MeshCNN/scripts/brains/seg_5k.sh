#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/native_wm_05k_right \
--name n_wm_5k_right \
--dataset_mode segmentation \
--arch meshunet \
--ninput_edges 15000 \
--ncf 32 64 \
--pool_res 4000 \
--batch_size 1 \
--lr 0.001 \

