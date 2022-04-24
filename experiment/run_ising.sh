#!/bin/bash


for p in 50 100 150 200;
do
  python fhmm.py \
  --p $p \
  --mu 2 \
  --sigma 3 \
  --lamda 1 \
  --n_steps 40000
done