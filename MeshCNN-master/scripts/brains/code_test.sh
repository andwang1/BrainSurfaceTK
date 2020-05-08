 #!/usr/bin/env bash
  ## run the training
  python3 train.py \
   --dataroot datasets/brains_testing \
   --checkpoints_dir checkpoints/red50_birth_age1 \
   --export_folder checkpoints/mesh_collapses \
   --name brains \
   --epoch_count 1 \
   --norm group \
   --num_aug 1 \
   --verbose_plot \
   --print_freq 10 \
   --seed 0 \
   --dataset_mode regression \
   --label birth_age \
   --features scan_age \
   --niter 5 \
   --niter_decay 100 \
   --batch_size 1 \
   --ncf 64 112 128 \
   --pool_res 4000 3500 3000 \
   --ninput_edges 9000 \
   --verbose_plot \

