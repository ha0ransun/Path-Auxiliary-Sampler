#!/bin/bash

dataset=dynamic_mnist

data_dir=$HOME/data/sip
sampling_steps=50
sampler=gwg-5

save_dir=$HOME/results/$sampler/$dataset-s-$sampling_steps

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python -m SIP.debm.experiment.main_binary_ebm \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --dataset_name $dataset \
    --sampling_steps $sampling_steps \
    --model resnet-64 \
    --buffer_size 10000 \
    --warmup_iters 10000 \
    --learning_rate 1e-4 \
    --n_iters 50000 \
    --buffer_init mean \
    --base_dist \
    --sampler $sampler \
    --eval_sampling_steps 10000 \
    --gpu 0 \
    $@
