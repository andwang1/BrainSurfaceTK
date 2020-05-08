#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_reg_native_10k \
--checkpoints_dir checkpoints/native_10k_lr_plateau_larger \
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
--ncf 128 128 256 512 \
--pool_res 3250 3000 2750 2500 \
--lr 0.0003 \
--init_type kaiming \
--lr_policy plateau \
--num_groups 2 \
--min_lr 1e-5 \
