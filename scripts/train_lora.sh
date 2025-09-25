#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

python ./train.py --dataset rotten_tomatoes --batch_size 1 --num_epoch 1 --model_path meta-llama/Meta-Llama-3.1-8B --train_method lora --lora_r 256 --save_every $1
rsync -ar ./finetune/ "$(rsync-path /home/ivo_petrov)"/dager/finetune
