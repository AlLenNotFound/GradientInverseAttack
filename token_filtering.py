import argparse
import sys
import os
import torch
import numpy as np
from utils.data import TextDataset
from utils.models import ModelWrapper

np.random.seed(102)
torch.manual_seed(102)

parser = argparse.ArgumentParser(description='DAGER attack')
parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
parser.add_argument('--label', type=str, default='name of the run', required=False)
argv = sys.argv[1:]
args=parser.parse_args(argv)
if args.neptune is not None:
    import neptune.new as neptune
    assert('label' in args)
    nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] , 'api_token':os.environ['HF_TOKEN']} 
    run = neptune.init( **nep_par )
    args_dict = vars(args)
    run[f"parameters"] = args_dict
    args.neptune = run

def check_if_in_span(R_K_norm, v):
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v ) # ( (R_K_norm @ v.T) [:,:,None] * R_K_norm[:,None,:] ).sum(0)
    out_of_span = proj - v
    size = out_of_span.pow(2).sum(-1).sqrt()
    return size

def filter_in_span(R_K_norm, v, thresh):
    size = check_if_in_span(R_K_norm, v)
    bools = size < thresh
    return torch.where( bools )

def get_top_B_in_span(R_K_norm, v, B, thresh):
    size = check_if_in_span(R_K_norm, v)
    bools = size < thresh
    which = torch.where( bools )
    _, idx = torch.sort( size[which] )
    which_new = []
    for w in which:
        which_new.append( w[idx] )
    which_new = tuple( which_new )
    return which_new

class CustomArgs:
    model_path='gpt2'
    cache_dir=None
    finetuned_path=None
    task='seq_class'
    grad_b=None
    device='cpu'
    pad='right'
    rank_cutoff=20

def log_n_tokens(l1_span_thresh, l2_span_thresh, l1_ntokens, l2_ntokens):
    print('========================================')
    print(f'L1 Rank Threshold: {l1_span_thresh} , L2 Rank Threshold: {l2_span_thresh}')
    print(f'Number of tokens after L1 filter: {l1_ntokens}')
    print(f'Number of tokens after L2 filter: {l2_ntokens}')
    print('========================================')
    if args.neptune:
        args.neptune['logs/l1_span_thresh'].log(l1_span_thresh)
        args.neptune['logs/l2_span_thresh'].log(l2_span_thresh)
        args.neptune['logs/l1_ntokens'].log(l1_ntokens)
        args.neptune['logs/l2_ntokens'].log(l2_ntokens)
def main():
    device = torch.device("cpu")
    dataset = TextDataset("cpu", "sst2", "val", 1, 32, None)

    model_wrapper = ModelWrapper(CustomArgs())
    
    sample = dataset[0] # (seqs, labels)

    print('reference: ', flush=True)
    for seq in sample[0]:
        print('========================')
        print(seq)
    print('========================')
    
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    
    sequences, true_labels = sample
    sequences = [' '.join(s.split(' ')[1:]) for s in sequences]
    orig_batch = tokenizer(sequences,padding=True, truncation=True, max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),return_tensors='pt')
    true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
    
    with torch.no_grad():
        B, R_Q, R_Q2 = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=1e-8)
        print(B)
        del true_grads 
               
        sentence_ends = []
        p = 0
        n_tokens = 0
        embeds = model_wrapper.get_embeddings(0)
        for l1_span_thresh in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
            _, res_ids = get_top_B_in_span(R_Q, embeds, 32, l1_span_thresh)
            if len(res_ids) == 0:
                for l2_span_thresh in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
                    log_n_tokens(l1_span_thresh, l2_span_thresh, 0, 0)
                continue
            res_pos = torch.ones_like( res_ids ) * p
            res_ids = torch.tensor(res_ids).unsqueeze(0).T
            seq_batch = res_ids
            attention_mask = torch.where(seq_batch != model_wrapper.pad_token, 1, 0)
            input_layer1 = model_wrapper.get_l1_output(seq_batch, attention_mask = attention_mask)
            sizesq2 = check_if_in_span(R_Q2, input_layer1)
            for l2_span_thresh in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
                l2_ntokens = (sizesq2 < l2_span_thresh).int().sum()
                log_n_tokens(l1_span_thresh, l2_span_thresh, res_ids.shape[0], l2_ntokens)

            
    
if __name__=='__main__':
    main()
