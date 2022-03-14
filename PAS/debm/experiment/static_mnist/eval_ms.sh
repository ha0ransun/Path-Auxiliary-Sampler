#!/bin/bash

dataset=static_mnist

data_dir=$HOME/data/sip
sampling_steps=30
ms_radius=7

sampler=mscorrect-$ms_radius
save_dir=$HOME/results/$sampler/$dataset-s-$sampling_steps

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=5

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
    --eval_sampling_steps 300000 \
    --eval_only \
    --model_dump best_ckpt.pt \
    --gpu 0 \
    $@
