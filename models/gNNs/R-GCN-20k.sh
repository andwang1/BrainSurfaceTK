#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnw119
whoami
source /vol/bitbucket/cnw119/miniconda3/etc/profile.d/conda.sh
conda activate vortexAI
# source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
cd /vol/bitbucket/cnw119/neodeepbrain || exit

python -u -m models.gNNs.basicgcntrain right 20k False some --batch_size 32 --save_path ../tmp2 --results ./results2
