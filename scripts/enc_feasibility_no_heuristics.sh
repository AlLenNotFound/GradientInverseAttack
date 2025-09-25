#!/bin/bash
python attack_len_increment.py --model_path bert-base-uncased --start_input 1 --end_input $2 --n_inputs 100 -b $1 --dataset glnmario/ECHR --rank_tol 4e-8 --label feasibility_encoders_no_heuristics_b$1 --parallel 1000 --l1_span_thresh 1e-4 --task seq_class --split val --l1_filter all --l2_filter  overlap --cache_dir ./models_cache
