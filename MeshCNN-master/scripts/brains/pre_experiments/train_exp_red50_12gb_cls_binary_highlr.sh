#!/usr/bin/env bash

## run the training
python3 train.py \
--dataroot datasets/brains_cls_binary_preterm_red50 \
--checkpoints_dir checkpoints/red50_12gb_cls_binary_highlr \
--export_folder checkpoints/mesh_collapses \
--name brains \
--ninput_edges 48735 \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--seed 0 \
--dataset_mode binary_class \
--niter 20 \
--niter_decay 80 \
--batch_size 2 \
--ncf 64 112 128 \
--pool_res 3000 2750 2500 \
--init_type kaiming \
--lr 0.002 \

