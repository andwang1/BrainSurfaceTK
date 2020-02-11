#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/brains_reduced \
--name brains \
--ncf 64 128 256 256 \
--pool_res 6000 4500 3000 1800 \
--norm group \
--resblocks 1 \
--export_folder meshes \
