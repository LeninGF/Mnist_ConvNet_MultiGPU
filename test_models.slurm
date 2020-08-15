#!/bin/bash

#SBATCH -J Mnist
#SBATCH -n 28
#SBATCH --gres=gpu:1
#SBATCH -p quinde-G9-2680
#SBATCH -w apollo-1
#SBATCH --qos=extended_prio

echo 'Loading models and evaluate performance'
start=$(date +'%s')
python load_and_test.py
end=$(date +'%s')
elapsed=$(($end - $start))
echo "Testing Mnist on TestSet took: $elapsed seconds"
