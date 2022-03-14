#!/bin/bash

model=lattice_ising_2d
dim=40
theta=0.5

export CUDA_VISIBLE_DEVICES=3

save_dir=$HOME/results/ising_learning/$model-$dim-$theta

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_ising_learning.py \
    --model $model \
    --save_dir $save_dir \
    --data_model lattice_ising \
    --dim $dim \
    --sigma $theta \
    --n_samples 2000 \
    --gt_steps 1000000 \
    --gpu 0 \
