#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:2:$len}

python attack.py --dataset $1 --split val --n_inputs 100 --batch_size $2 --l1_filter maxB --l2_filter non-overlap --model_path bert-base-uncased --device cuda --task seq_class --cache_dir ./models_cache $last_args
