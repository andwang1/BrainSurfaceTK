#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnw119
whoami

source /vol/bitbucket/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate vortexAI

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
cd /vol/bitbucket/${USER}/neodeepbrain || exit
python -u -m models.gNNs.segmentationbasicgcntrain
