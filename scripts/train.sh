#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:1:$len}

python3 train.py --dataset rotten_tomatoes --batch\_size 32 --noise $1 --num\_epochs 2 --save\_every 100 --model\_path gpt2
