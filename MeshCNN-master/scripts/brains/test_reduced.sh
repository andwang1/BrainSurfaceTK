#!/usr/bin/env bash

## run the test and export collapses
python3 test.py \
--dataroot datasets/brains_reduced_90_gender \
--export_folder checkpoints/mesh_collapses \
--name brains \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--export_folder meshes \
--dataset_mode regression \
--verbose_plot \
--results_dir results/regression_results \
