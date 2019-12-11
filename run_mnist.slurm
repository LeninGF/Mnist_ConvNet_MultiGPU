#!/bin/bash

#SBATCH -J WR-Inb
#SBATCH -n 28
#SBATCH --gres=gpu:2
#SBATCH -p quinde-G9-2680
#SBATCH -w apollo-2
#SBATCH --qos=extended

echo 'Training Mnist with 2 GPUS'
start=$(date +'%s')
python main.py
end=$(date +'%s')
elapsed=$(($end - $start))
echo "Training Mnist on 20 epochs took: $elapsed seconds"