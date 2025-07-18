#!/bin/bash
mkdir -p logs
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_sac
nohup tensorboard --logdir runs --port 4444 --bind_all >> logs/tensorboard.log 2>&1 &
