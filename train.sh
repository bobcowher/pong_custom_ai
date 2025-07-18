#!/bin/bash
mkdir -p logs
export MUJOCO_GL=egl
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pong 
nohup python ./train.py >> logs/train.log 2>&1 & 
