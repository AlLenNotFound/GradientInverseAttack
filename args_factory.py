import argparse
import time
import sys

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='DAGER attack')

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')
    parser.add_argument('--label', type=str, default='name of the run', required=False)
    
    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101) 
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', 'stanfordnlp/imdb', 'glnmario/ECHR'], required=True)
    parser.add_argument('--task', choices=['seq_class', 'next_token_pred'], required=True)
    parser.add_argument('--pad', choices=['right', 'left'], default='right')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True) # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--finetuned_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_grad', type=str, default='cpu')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['sdpa', 'eager'])

    parser.add_argument('--precision', type=str, default='full', choices=['8bit', 'half', 'full', 'double'])
    parser.add_argument('--parallel', type=int, default=1000)
    parser.add_argument('--grad_b', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--rank_tol', type=float, default=None) 
    parser.add_argument('--rank_cutoff', type=int, default=20)
    parser.add_argument('--l1_span_thresh', type=float, default=0.2)
    parser.add_argument('--l2_span_thresh', type=float, default=1e-7)
    parser.add_argument('--l1_filter', choices=['maxB', 'all'], required=True)
    parser.add_argument('--l2_filter', choices=['overlap', 'non-overlap'], required=True)
    parser.add_argument('--distinct_thresh', type=float, default=0.7)
    parser.add_argument('--max_ids', type=int, default=-1)
    parser.add_argument('--maxC', type=int, default=10000000) 
    parser.add_argument('--reduce_incorrect', type=int, default=0)
    parser.add_argument('--n_incorrect', type=int, default=None)

    parser.add_argument('--repeat_window', type=int, default=8)
    parser.add_argument('--repeat_penalty', type=float, default=0.8)
    parser.add_argument('--lm_rerank_lambda', type=float, default=0.0)

    parser.add_argument('--require_last_k', type=int, default=0)
    parser.add_argument('--l2_keep_ratio', type=float, default=0.95)
    parser.add_argument('--l2_span_auto', action='store_true')
    parser.add_argument('--l2_auto_quantile', type=float, default=0.25)

    parser.add_argument('--show_tqdm', action='store_true')  # 想看进度条时再开

    # FedAVG
    parser.add_argument('--algo', type=str, default='sgd', choices=['sgd', 'fedavg'])
    parser.add_argument('--avg_epochs', type=int, default=None)
    parser.add_argument('--b_mini', type=int, default=None)
    parser.add_argument('--avg_lr', type=float, default=None)
    parser.add_argument('--dist_norm', type=str, default='l2', choices=['l1', 'l2'])
    
    #DP
    parser.add_argument('--defense_noise', type=float, default=None) # add noise to true grads
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--p1_std_thrs', type=float, default=5)
    parser.add_argument('--l2_std_thrs', type=float, default=5)
    parser.add_argument('--dp_l2_filter', type=str, default='maxB', choices=['maxB', 'outliers'])
    parser.add_argument('--defense_pct_mask', type=float, default=None) # mask some percentage of gradients
    
    #Dropout
    parser.add_argument('--grad_mode', type=str, default='eval', choices=['eval', 'train'])
    
    #Rebuttal experiments
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])

    #LoRA
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)

    parser.add_argument('--headwise_factorization', action='store_true', help='Enable head-wise factorization for the first layer gradient.')


    if argv is None:
       argv = sys.argv[1:]
    args=parser.parse_args(argv)

    if args.n_incorrect is None:
        args.n_incorrect = args.batch_size

    if args.neptune is not None:
        import neptune.new as neptune
        assert('label' in args)
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] } 
        if args.neptune_offline:
            nep_par['mode'] = 'offline'
            args.neptune_id = 'DAG-0'

        run = neptune.init( **nep_par )
        args_dict = vars(args)
        run[f"parameters"] = args_dict
        args.neptune = run
        if not args.neptune_offline:
            print('waiting...')
            start_wait=time.time()
            args.neptune.wait()
            print('waited: ',time.time()-start_wait)
            args.neptune_id = args.neptune['sys/id'].fetch()
        print( '\n\n\nArgs:', *argv, '\n\n\n' ) 
    return args
