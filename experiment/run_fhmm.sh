#!/bin/bash


for L in 500 1000 1500 2000;
do
  python fhmm.py \
  --L $L \
  --alpha 0.05 \
  --beta 0.85 \
  --sigma 0.5 \
  --n_steps 40000 \
  --L 1000 \
  --K 10
done