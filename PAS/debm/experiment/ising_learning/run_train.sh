#!/bin/bash

model=lattice_ising_2d
dim=25
theta=0.25
sampler=gwg
steps=50

export CUDA_VISIBLE_DEVICES=5

save_root=$HOME/results/ising_learning/$model-$dim-$theta
save_dir=$save_root/$sampler-n-$steps

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_ising_learning.py \
    --model $model \
    --save_dir $save_dir \
    --data_model lattice_ising \
    --sampler $sampler \
    --data_file $save_root/data.pkl \
    --buffer_size 256 \
    --batch_size 256 \
    --rbm_lr 0.001 \
    --l1 0.01 \
    --n_iters 100000 \
    --dim $dim \
    --sigma $theta \
    --n_samples 2000 \
    --sampling_steps $steps \
    --gt_steps 1000000 \
    --gpu 0 \
