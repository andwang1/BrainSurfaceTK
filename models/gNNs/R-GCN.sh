#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnw119
whoami
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
conda activate vortexAI
# source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
cd /vol/bitbucket/${USER}/neodeepbrain || exit

python -u -m models.gNNs.basicgcntrain right 10k True
python -u -m models.gNNs.basicgcntrain right 10k False
python -u -m models.gNNs.basicgcntrain right 20k True --batch_size 32
python -u -m models.gNNs.basicgcntrain right 20k False --batch_size 32
