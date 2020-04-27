#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_merged_10k \
--checkpoints_dir checkpoints/merged_10k_lr_cosine_larger \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode regression \
--niter 1 \
--niter_decay 300 \
--batch_size 1 \
--ncf 256 256 512 1024 \
--pool_res 3500 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \
--lr_policy cosine_restarts \
--lr_decay_iters 10 \
--min_lr 1e-5 \
