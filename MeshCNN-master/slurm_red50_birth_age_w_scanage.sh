#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aw1912
export PATH=/vol/project/2019/545/g1954504/Andy/deepl_brain_surfaces/MeshCNN-master/:$PATH
source /vol/biomedic2/aa16914/shared/MScAI_brain_surface/venv/bin/activate
TERM=vt100 # or TERM=xterm
source /vol/cuda/10.0.130/setup.sh
/usr/bin/nvidia-smi
uptime
pwd
/bin/bash ./scripts/brains/train_exp_red50_12gb_birthage_w_scanage.sh
