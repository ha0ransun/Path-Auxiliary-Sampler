#!/bin/bash

dataset=omniglot

data_dir=$HOME/data/sip
sampling_steps=50
ms_radius=5

sampler=mscorrect-$ms_radius
save_dir=$HOME/results/$sampler/$dataset-s-$sampling_steps

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=4


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
    --plot_every 4 \
    --eval_every 21 \
    --base_dist \
    --sampler $sampler \
    --eval_only \
    --model_dump best_ckpt.pt \
    --eval_sampling_steps 300000 \
    --gpu 0 \
    $@
