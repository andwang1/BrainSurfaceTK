#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnw119
export PATH=/vol/biomedic2/aa16914/shared/MScAI_brain_surface/venv/bin/:$PATH
source activate
source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
bash /vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/MeshCNN-master/scripts/brains/train.sh
