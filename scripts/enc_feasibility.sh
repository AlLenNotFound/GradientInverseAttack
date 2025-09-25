#!/bin/bash

python attack_len_increment.py --model_path bert-base-uncased --n_inputs 100 -b $1 --dataset glnmario/ECHR --rank_tol 1e-9 --label feasibility_encoders_heuristics_$1 --parallel 1000 --start_input 1 --end_input $2 --l1_span_thresh 1e-4 --task seq_class --split val --l1_filter maxB --l2_filter  non-overlap --cache_dir ./models_cache
