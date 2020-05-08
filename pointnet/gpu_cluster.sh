#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=az519
export PATH=/vol/project/2019/545/g1954504/alex/deepl_brain_surfaces/:$PATH
source /vol/project/2019/545/g1954504/venv_1.2/bin/activate
TERM=vt100 # or TERM=xterm
source /vol/cuda/10.0.130/setup.sh
/usr/bin/nvidia-smi
uptime
pwd
python3 -m main.pointnet2-segmentation