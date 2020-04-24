#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_native_10k \
--checkpoints_dir checkpoints/batch_native_10k_lr_cosine \
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
--niter_decay 100 \
--batch_size 32 \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--num_groups 2 \
--lr_policy cosine_restarts \
--lr_decay_iters 10 \
--min_lr 1e-5 \
