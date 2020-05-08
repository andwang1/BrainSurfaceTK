#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_merged_10k \
--checkpoints_dir checkpoints/cls_gender_merged_10k_lr_static \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode binary_class \
--niter 200 \
--niter_decay 0 \
--batch_size 1 \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \