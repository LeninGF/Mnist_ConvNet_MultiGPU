#!/bin/bash

#SBATCH -J WR-Inb
#SBATCH -n 28
#SBATCH --gres=gpu:1
#SBATCH -p quinde-G9-2680
#SBATCH -w apollo-2
#SBATCH --qos=extended

echo 'Loading models and evaluate performance'
start=$(date +'%s')
python load_and_test.py
end=$(date +'%s')
elapsed=$(($end - $start))
echo "Testing Mnist on TestSet took: $elapsed seconds"