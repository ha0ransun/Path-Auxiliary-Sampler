#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

data_dir=$HOME/data/sip
model_dump=mnist_1234567

save_root=results
save_dir=$save_root/$model_dump

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main_rbm.py \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --seed 0 \
    --n_hidden 500 \
    --plot_every 100 \
    --gpu 0 \
    --model_dump $save_root/${model_dump}.ckpt \
