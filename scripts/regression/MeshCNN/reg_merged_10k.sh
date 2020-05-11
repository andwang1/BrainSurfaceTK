#!/usr/bin/env bash

## run the training
pushd models/MeshCNN

python3 train.py \
--dataroot datasets/brains \
--checkpoints_dir checkpoints/reg \
--export_folder checkpoints/mesh_collapses \
--name brains \
--epoch_count 1 \
--norm group \
--num_aug 1 \
--verbose_plot \
--print_freq 10 \
--dataset_mode regression \
--label scan_age \
--niter 1 \
--niter_decay 100 \
--batch_size 1 \
--ncf 16 32 \
--pool_res 3000 2750 \
--lr 0.0003 \
--init_type kaiming \
--lr_policy plateau \
--num_groups 2 \
--min_lr 3e-5 \

popd
