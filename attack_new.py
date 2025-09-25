import copy
import datetime
import itertools
import numpy as np
import torch
from datasets import load_metric
from utils.models import remove_padding, ModelWrapper
from utils.data import TextDataset
from args_factory import get_args
import time

from scipy.optimize import linear_sum_assignment

import itertools
from tqdm import tqdm
# old seed: 100
args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0

def check_if_in_span(R_K_norm, v, norm='l2'):
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v ) # ( (R_K_norm @ v.T) [:,:,None] * R_K_norm[:,None,:] ).sum(0)
    out_of_span = proj - v
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)

    return size

def filter_in_span(R_K_norm, v, thresh):
    size = check_if_in_span(R_K_norm, v)
    bools = size < thresh
    return torch.where( bools )

def get_top_B_in_span(R_K_norm, v, B, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)
    bools = size < thresh
    which = torch.where( bools )
    _, idx = torch.sort( size[which] )
    which_new = []
    for w in which:
        which_new.append( w[idx] )
    which_new = tuple( which_new )
    return which_new

def filter(args, model_wrapper, R_Qs, l, token_type, res_ids, sentence_filter, approx_sentence_filter, approx_sentence_score, max_ids, B):
    predicted_sentences = [ [-1]*(l+1) for i in range(B) ]
    predicted_sentences_scores = [ torch.inf for i in range(B) ]
    if args.n_layers > 2:
        raise NotImplementedError()
    else:
        R_Q2 = R_Qs[1]
    res_ids = res_ids.copy()
    for i in range(l):
        if max_ids >= 0:
            res_ids[i] = res_ids[i][:min(max_ids, len(res_ids[i]))]

    lens_array = []
    max_lens = -1

    for s in sentence_filter:
        for i, t in enumerate( s ):
            if t != model_wrapper.start_token and len(res_ids[i]) > 1 and t in res_ids[i]:
                token_ind = res_ids[i].index(t)
                res_ids[i] = res_ids[i][:token_ind] + res_ids[i][token_ind+1:]

    total_num_combos = 1
    for i, res_id in enumerate(res_ids):
        if i >= l:
            break
        lens = len(res_id)
        total_num_combos *= lens
        lens_array.append( list(range(lens)) )
        if max_lens < lens:
            max_lens = lens
    lst = itertools.product(*lens_array)

    res_ids_final = torch.tensor(np.array( [ res_id + [-1]*(max_lens - len(res_id))  for i, res_id in enumerate(res_ids) if i < l ] )).to(args.device)
    
    it_lst = iter( lst )
    print( f'Len {l}:{total_num_combos}' )
    sizesq2_bad_best = None
    best_bad_sentence = None
    best_bad_sentence_words = None
    progress_bar = tqdm(total=total_num_combos)
    passed = 0
    while passed < args.maxC:
        if total_num_combos < args.maxC:
            els = []
            for i in range(args.parallel):
                el = next(it_lst, None)
                if el is None:
                    break
                els.append(el)
            els = torch.tensor(np.array(els)).to(args.device)
            if els.shape[0] == 0:
                break
        else:
            els = []
            for token_num, tokens in  enumerate(res_ids):
                if token_num >= l:
                    break
                els.append( torch.randint(len(tokens), (args.parallel,1)) )
            els = torch.cat( els, axis=1 ).to(args.device)
        passed += els.shape[0]

        sentences = res_ids_final[torch.arange(res_ids_final.shape[0]), els]
        sentences = torch.concat((sentences, torch.tensor([model_wrapper.eos_token]).reshape(1,1).repeat(els.shape[0],1).to(args.device)),dim=1)
        
        if model_wrapper.is_bert():
            token_type_ids = torch.ones_like( sentences ) * token_type
            input_layer1 = model_wrapper.get_layer_inputs(sentences, token_type_ids)[0]
        else:
            input_layer1 = model_wrapper.get_layer_inputs(sentences)[0]
            
        
        sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm).mean(dim=1)
        
        # Remove repeated versions of approximate sentences
        for b_idx in range(len(approx_sentence_filter)):
            ext_sent = torch.tensor(approx_sentence_filter[b_idx] + [-1]*(l+1 - len(approx_sentence_filter[b_idx]))).to(args.device)
            already_predicted = (ext_sent == sentences).sum(1) >= (l+1)*args.distinct_thresh
            better_score = sizesq2 > approx_sentence_score[b_idx]
            sizesq2[ torch.logical_and( already_predicted, better_score ) ] = torch.inf

        # Remove better versions of existing senteces
        for b_idx in range(B):
            already_predicted = (torch.tensor(predicted_sentences[b_idx]).to(args.device) == sentences).sum(1) >= (l+1)*args.distinct_thresh
            better_score = sizesq2 > predicted_sentences_scores[b_idx]
            sizesq2[ torch.logical_and( already_predicted, better_score ) ] = torch.inf

        # Draw unique
        scores_best_batch, sentences_best_batch = [], [] 
        for b_idx in range(B):
            idx_best_batch = torch.argmin(sizesq2)
            best_score = sizesq2[idx_best_batch]
            best_sentence = sentences[idx_best_batch]
            sentences_best_batch.append( best_sentence.tolist() )
            scores_best_batch.append( best_score.item() )
            similar_sentences = (best_sentence == sentences).sum(1) >= (l+1)*args.distinct_thresh
            sizesq2[similar_sentences] = torch.inf

        # Swap out better batch options
        for b_idx in range(len(scores_best_batch)):
            if scores_best_batch[b_idx] > predicted_sentences_scores[-1]:
                break
            predicted_idx = 0
            while scores_best_batch[b_idx] > predicted_sentences_scores[predicted_idx]:
                predicted_idx += 1
            predicted_sentences_scores = predicted_sentences_scores[:-1]
            predicted_sentences = predicted_sentences[:-1]
            predicted_sentences = predicted_sentences[:predicted_idx] + sentences_best_batch[b_idx:b_idx+1] + predicted_sentences[predicted_idx:]
            predicted_sentences_scores = predicted_sentences_scores[:predicted_idx] + scores_best_batch[b_idx:b_idx+1] + predicted_sentences_scores[predicted_idx:]

        progress_bar.update( els.shape[0] )
    progress_bar.close()
    return predicted_sentences, predicted_sentences_scores 

def filter_outliers(model_wrapper, d, stage='token', std_thrs=None, maxB=None):
    if std_thrs is None:
        res_ids = torch.tensor(d.argsort()[:maxB])
        bools = torch.zeros_like(d).bool()
        bools[res_ids] = True
    elif maxB is None:
        print(f'Wrong dists: {d.mean()} +- {d.std()}')
        d = (d - d.mean())/d.std()
        bools = d < -std_thrs
        res_ids = torch.tensor(np.nonzero(bools)[:, 0])
    else:
        bools = torch.zeros_like(d).bool()
        bools[torch.tensor(d.argsort()[:maxB])] = True
        print(f'Wrong dists: {d.mean()} +- {d.std()}')
        d = (d - d.mean())/d.std()
        bools = bools & (d < -std_thrs)
        res_ids = torch.tensor(np.nonzero(bools)[:, 0])
    
    if stage=='token':
        return res_ids
    else:
        return torch.tensor(d).unsqueeze(1), torch.tensor(bools)

def get_span_dists(model_wrapper, R_Qs, embeds, p=0, stage='token'):
    dists = []
    if stage == 'token':
        dists.append(check_if_in_span(R_Qs[0], embeds, args.dist_norm).T)
        sentences = torch.arange(embeds.shape[1]).unsqueeze(1).to(model_wrapper.args.device)
        embs = model_wrapper.get_layer_inputs(sentences, layers=args.n_layers-1)
        
    else:
        embs = [e.to(model_wrapper.args.device) for e in embeds]
        
    if p==0:
        for i in range(model_wrapper.args.n_layers-1):
                dists.append(check_if_in_span(R_Qs[i+1], embs[i], args.dist_norm))
    
    print('dists', torch.cat(dists, axis=1).shape)
    d = torch.log(torch.cat(dists, axis=1))-torch.log(1-torch.cat(dists, axis=1))
    d = d.mean(axis=1).cpu().detach()
    
    return d
def filter_l1(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
        
    sentence_ends = []
    p = 0
    n_tokens = 0

    while True:
        print(f'L1 Position {p}')
        embeds = model_wrapper.get_embeddings(p)
        if model_wrapper.is_bert():
            if args.defense_noise is None:
                _, res_ids_new, res_types_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
            else:
                raise NotImplementedError
        else:
            if args.defense_noise is None:
                _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, args.batch_size, args.l1_span_thresh, args.dist_norm)
            else:
                std_thrs = args.p1_std_thrs if p==0 else None
                d = get_span_dists(model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(model_wrapper, d, std_thrs=std_thrs, maxB=max(50*model_wrapper.args.batch_size, int(0.05*len(model_wrapper.tokenizer))))
            res_types_new = torch.zeros_like(res_ids_new)
        res_pos_new = torch.ones_like( res_ids_new ) * p
        
        del embeds
        
        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > tokenizer.model_max_length or p > args.max_len:
            break
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][ end_token_ind ]
            sentence_ends.append((p,sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind+1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind+1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
        if model_wrapper.has_rope():
            break
        
    return res_pos, res_ids, res_types, sentence_ends

def filter_decoder_step(args, model_wrapper, R_Qs, batch, p):
    if args.defense_noise is None:
        R_Q2 = R_Qs[1]
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layer1 = model_wrapper.get_layer_inputs(batch, attention_mask = attention_mask)[0]
        sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
        boolsq2 = sizesq2 < args.l2_span_thresh
        
        if model_wrapper.has_rope():
            boolsq2 = torch.logical_or(boolsq2, torch.isin(batch, torch.tensor([model_wrapper.pad_token, model_wrapper.start_token]).to(args.device)))
            if p>1:
                repeats = torch.logical_and(batch[:, -2]==model_wrapper.start_token, torch.isin(batch[:, -1], batch[:, 1].to(args.device)))
                correct_sentences = torch.logical_and(boolsq2.all(dim=1), ~repeats.to(args.device))
            else:
                correct_sentences = boolsq2.all(dim=1)
        else:
            correct_sentences = boolsq2.all(dim=1)
                
        return sizesq2, correct_sentences
    
    else:
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layers = model_wrapper.get_layer_inputs(batch, attention_mask = attention_mask, layers=args.n_layers-1)
        return get_span_dists(model_wrapper, R_Qs, input_layers, stage='sequence')

def filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=-1):
    R_Q2 = R_Qs[1]
    res_ids = copy.deepcopy(res_ids)
    for i in range(len(res_ids)):
        if max_ids >= 0:
            res_ids[i] = res_ids[i][:min(max_ids, len(res_ids[i]))]
    if args.pad == 'right':
        batch = torch.tensor(res_ids[0]).unsqueeze(1)
    elif args.pad == 'left':
        start_ids = res_ids[0].copy()
        if model_wrapper.start_token is not None:
            start_ids = [model_wrapper.start_token]
        batch = torch.tensor(start_ids).unsqueeze(1)
        
    is_batch_incorrect = torch.zeros_like(batch).squeeze(1)

    scores = check_if_in_span(R_Q2, model_wrapper.get_layer_inputs(batch.to(args.device))[0], args.dist_norm).mean(dim=1).to('cpu')

    predicted_sentences = []
    predicted_sentences_scores = []
    
    top_B_incorrect_sentences = [[] for i in range(args.batch_size)]
    top_B_incorrect_scores = [torch.inf for i in range(args.batch_size)]
    
    i = 1
    while True:
        print(f'Position {i}')

        top_B_incorrect_sentences_len = [[] for i in range(args.batch_size)]
        top_B_incorrect_scores_len = [torch.inf for i in range(args.batch_size)]
        
        if len(batch) == 0 or (not model_wrapper.has_rope() and i >= len(res_ids)):
            break
        
        if model_wrapper.has_rope():
            ends = torch.Tensor(res_ids[0])
            ends = ends[ends != model_wrapper.pad_token]
        else:
            ends = torch.Tensor(res_ids[i])
        
        lst = itertools.product(range(batch.shape[0]), range(len(ends)))
        it_lst = iter(lst)
        next_batch = []
        next_scores = []
        is_next_batch_incorrect = []
        ds = []
        is_complete=args.defense_noise is None
        curr_sentence = 0
        progress_bar = tqdm(total=batch.shape[0]*ends.shape[0])
        
        while True:
            els_b = []
            els_ends = []
            for _ in range(max((args.parallel//ends.shape[0]), 1)*ends.shape[0]):
                el = next(it_lst, None)
                if el is None:
                    break
                els_b.append(el[0])
                els_ends.append(el[1])
            els_b = torch.tensor(np.array(els_b))
            els_ends = torch.tensor(np.array(els_ends))
            if els_b.shape[0] == 0 and is_complete:
                break
            elif els_b.shape[0] == 0:
                idxs = np.array(list(itertools.product(range(batch.shape[0]), range(len(ends)))))
                new_batch = torch.cat((torch.tensor(batch[idxs[:, 0]]).long(),\
                                       torch.tensor(ends[idxs[:, 1]]).long().unsqueeze(1)), dim=-1).to(args.device)
                is_new_batch_incorrect = is_batch_incorrect[idxs[:, 0]].to(args.device)
                sizesq2 = torch.cat(ds)
                sizesq2, correct_sentences = filter_outliers(model_wrapper, sizesq2, stage='sequence', std_thrs=args.l2_std_thrs, maxB=args.batch_size)
                is_complete = True
                print(sizesq2.min())
            else:
                new_batch = torch.cat((batch[els_b], ends[els_ends].unsqueeze(1)),dim=-1).int().to(args.device)
                is_new_batch_incorrect = is_batch_incorrect[els_b].to(args.device)
                
                if args.defense_noise is None:
                    sizesq2, correct_sentences = filter_decoder_step(args, model_wrapper, R_Qs, new_batch, i)
                else:
                    ds.append(filter_decoder_step(args, model_wrapper, R_Qs, new_batch, i))
                    continue           
            
            
            if i > 1:
                complete_batches = torch.where(~correct_sentences.reshape(-1, ends.shape[0]).any(dim=1))[0]
                for pred_idx in complete_batches:
                    if not is_batch_incorrect[curr_sentence+pred_idx]:
                        predicted_sentences.append(batch[curr_sentence+pred_idx].cpu().numpy().tolist())
                        predicted_sentences_scores.append(scores[curr_sentence+pred_idx].item())
                    
            next_batch.append(new_batch[correct_sentences].to('cpu'))
            if model_wrapper.has_bos():
                next_scores.append(sizesq2[:, 1:].mean(dim=1)[correct_sentences].to('cpu'))
            else:
                next_scores.append(sizesq2.mean(dim=1)[correct_sentences].to('cpu'))
            is_next_batch_incorrect.append(is_new_batch_incorrect[correct_sentences].to('cpu'))
            
            curr_sentence += len(els_b)//ends.shape[0]
                
            incorrect_sentences = new_batch[~correct_sentences]
            if model_wrapper.has_bos():
                sizesq2_incorrect = sizesq2[~correct_sentences, 1:].mean(dim=1)
            else:
                sizesq2_incorrect = sizesq2[~correct_sentences].mean(dim=1)

            if len(incorrect_sentences) == 0:
                continue
            
            # Draw unique
            scores_best_batch, sentences_best_batch = [], [] 
            for b_idx in range(args.n_incorrect):
                idx_best_batch = torch.argmin(sizesq2_incorrect)
                best_score = sizesq2_incorrect[idx_best_batch]
                best_sentence = incorrect_sentences[idx_best_batch]
                sentences_best_batch.append( best_sentence.cpu().numpy().tolist() )
                scores_best_batch.append( best_score.item() )
                similar_sentences = (best_sentence == incorrect_sentences).sum(1) >= (i+1)*args.distinct_thresh
                sizesq2_incorrect[similar_sentences] = torch.inf

            for b_idx in range(len(scores_best_batch)):
                if scores_best_batch[b_idx] > top_B_incorrect_scores_len[-1]:
                    break
                predicted_idx = 0
                while scores_best_batch[b_idx] > top_B_incorrect_scores_len[predicted_idx]:
                    predicted_idx += 1
                for rep_idx in range(predicted_idx, args.n_incorrect):
                    if len(top_B_incorrect_sentences_len[rep_idx]) > 0 and\
                        (torch.tensor(sentences_best_batch[b_idx:b_idx+1]) == torch.tensor(top_B_incorrect_sentences_len[rep_idx:rep_idx+1])).sum(1) \
                        >= (i+1)*args.distinct_thresh:
                            
                        continue
                    else:
                        top_B_incorrect_sentences_len = top_B_incorrect_sentences_len[:predicted_idx] + sentences_best_batch[b_idx:b_idx+1] + top_B_incorrect_sentences_len[predicted_idx:rep_idx] +top_B_incorrect_sentences_len[rep_idx+1:]
                        top_B_incorrect_scores_len = top_B_incorrect_scores_len[:predicted_idx] + scores_best_batch[b_idx:b_idx+1] + top_B_incorrect_scores_len[predicted_idx:rep_idx] + top_B_incorrect_scores_len[rep_idx+1:]
                        break
            progress_bar.update(new_batch.shape[0])
        
        batch = torch.cat(next_batch)
        if len(batch) == 0:
            break
        is_batch_incorrect = torch.cat(is_next_batch_incorrect)
        scores = torch.cat(next_scores)
        if i != len(res_ids) - 1 and len(top_B_incorrect_sentences_len[0]) > 0:
            batch = torch.cat((batch, torch.tensor(top_B_incorrect_sentences_len)))
            scores = torch.cat((scores, torch.tensor(top_B_incorrect_scores_len)))
            is_batch_incorrect = torch.cat((is_batch_incorrect, torch.ones(len(top_B_incorrect_sentences_len))))

        top_B_incorrect_scores += top_B_incorrect_scores_len
        top_B_incorrect_sentences += top_B_incorrect_sentences_len
        
        if args.reduce_incorrect > 0:
            final_incorrect_scores = []
            final_incorrect_sentences = []
            sorted_idx = np.argsort(top_B_incorrect_scores)[::-1]
            for j, idx in enumerate(sorted_idx):
                if len(top_B_incorrect_scores) - j <= args.batch_size - len(final_incorrect_scores):
                    final_incorrect_scores.append(top_B_incorrect_scores[idx])
                    final_incorrect_sentences.append(top_B_incorrect_sentences[idx])
                    continue
                proposal_sent = np.array(top_B_incorrect_sentences[idx])
                fail = False
                for accepted_sent in final_incorrect_sentences:
                    if len(accepted_sent) < len(proposal_sent):
                        s1 = np.pad(accepted_sent, (0, len(proposal_sent) - len(accepted_sent) ), 'constant', constant_values=(0, -1))
                        s2 = proposal_sent
                    else:
                        s1 = np.pad(proposal_sent, (0, len(accepted_sent) - len(proposal_sent) ), 'constant', constant_values=(0, -1))
                        s2 = accepted_sent
                    if np.sum(s1 == s2) < len(s1)*args.distinct_thresh:
                        fail = True
                        break
                if not fail:
                    final_incorrect_scores.append(top_B_incorrect_scores[idx])
                    final_incorrect_sentences.append(top_B_incorrect_sentences[idx])
            top_B_incorrect_scores = final_incorrect_scores
            top_B_incorrect_sentences = final_incorrect_sentences  
        progress_bar.close()
        i += 1
    # Add remaining sentences
    for i in range(batch.shape[0]):
        predicted_sentences.append(batch[i].cpu().numpy().tolist())
        predicted_sentences_scores.append(scores[i].item())
              
    return predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores

def reconstruct(args, sample, metric, model_wrapper):
    global total_correct_tokens, total_tokens, total_correct_maxB_tokens
    tokenizer = model_wrapper.tokenizer
    
    sequences, true_labels = sample
    
    orig_batch = tokenizer(sequences,padding=True, truncation=True, max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),return_tensors='pt').to(args.device)
    
    true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
    if args.defense_noise is not None:
        for grad in true_grads:
            grad.data = grad.data + torch.randn(grad.shape) * args.defense_noise
    prediction, predicted_sentences, predicted_sentences_scores = [], [], []

    with torch.no_grad():
        
        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range( orig_batch['input_ids'].shape[1] ):
            total_true_token_count2 += args.batch_size - ( orig_batch['input_ids'][:,i] == model_wrapper.pad_token).sum()
            uniques = torch.unique(orig_batch['input_ids'][:,i])
            total_true_token_count += uniques.numel()
            if model_wrapper.pad_token in uniques.tolist():
                total_true_token_count -= 1
        
        B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=100, tol=args.rank_tol)
        if B is None:
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]
            return ['' for _ in range(len(reference))], reference   
             
        R_Qs = [R_Q.to(args.device) for R_Q in R_Qs]
                
        print(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        if args.neptune:
            args.neptune['logs/max_rank'].log( B )
            args.neptune['logs/batch_tokens'].log( total_true_token_count2 ) 
            args.neptune['logs/batch_unique_tokens'].log( total_true_token_count )
         
        del true_grads 
       
        res_pos, res_ids, res_types, sentence_ends = filter_l1(args, model_wrapper, R_Qs)
        
        print( orig_batch )
        print( orig_batch['input_ids'].T )
        if len(res_ids) == 0:        
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]        
            return ['' for _ in reference], reference
        if len(res_ids[0])< 500:
            print( res_pos, res_ids, res_types )
        
        rec_l1, rec_l1_maxB, rec_l2 = [], [], []

        for s in range( orig_batch['input_ids'].shape[0] ):
            sentence_in = True
            sentence_in_max_B = True
            orig_sentence = orig_batch['input_ids'][s]
            last_idx = torch.where(orig_batch['input_ids'][s] != tokenizer.pad_token_id)[0][-1].item()
            for pos, token in enumerate( orig_sentence ):
                if not model_wrapper.is_bert() and pos == last_idx:
                    break
                if pos >= len(res_ids) and not model_wrapper.has_rope():
                    sentence_in = False
                    break
                if token == model_wrapper.pad_token and args.pad=='right':
                    pos-=1
                    break
                elif token == model_wrapper.pad_token and args.pad=='left':
                    continue
                if model_wrapper.has_rope():
                    total_correct_tokens += 1 if token in res_ids[0] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[0][:min(args.batch_size, len(res_ids[0]))] else 0
                    total_tokens += 1
                else:
                    total_correct_tokens += 1 if token in res_ids[pos] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))] else 0   
                    total_tokens += 1
                if token == model_wrapper.eos_token and args.pad=='right':
                    break
                
                if model_wrapper.has_rope():
                    if model_wrapper.has_bos() and token==model_wrapper.start_token:
                        continue
                    sentence_in = sentence_in and (token in res_ids[0])
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[0][:min(args.batch_size, len(res_ids[0]))])
                else:
                    sentence_in = sentence_in and (token in res_ids[pos]) 
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))]) 
            if model_wrapper.is_bert():
                sentence_in = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends
                sentence_in_max_B = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends

            rec_l1.append( sentence_in )
            rec_l1_maxB.append( sentence_in_max_B )
            if model_wrapper.has_rope():
                orig_sentence = (orig_sentence).reshape(1,-1)
            else:
                orig_sentence = (orig_sentence[:pos+1]).reshape(1,-1)
            if model_wrapper.is_bert():
                token_type_ids = (orig_batch['token_type_ids'][s][:orig_sentence.shape[1]]).reshape(1,-1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, token_type_ids)[0]
            else:
                attention_mask = orig_batch['attention_mask'][s][:orig_sentence.shape[1]].reshape(1,-1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, attention_mask=attention_mask)[0]
                
            sizesq2 = check_if_in_span(R_Qs[1], input_layer1, args.dist_norm)
            boolsq2 = sizesq2 < args.l2_span_thresh
            print( sizesq2 )
        
            if args.task == 'next_token_pred':
                rec_l2.append( torch.all(boolsq2[:-1]).item() )
            elif model_wrapper.has_rope(): 
                rec_l2.append( torch.all(boolsq2[1:]).item() )
            else:
                rec_l2.append( torch.all(boolsq2).item() )
        
        print( f'Rec L1: {rec_l1}, Rec L1 MaxB: {rec_l1_maxB}, Rec MaxB Token: {total_correct_maxB_tokens/total_tokens}, Rec Token: {total_correct_tokens/total_tokens}, Rec L2: {rec_l2}' )

        if args.neptune:
            args.neptune['logs/rec_l1'].log( np.array(rec_l1).sum() )
            args.neptune['logs/rec_l1_max_b'].log( np.array(rec_l1_maxB).sum() ) 
            args.neptune['logs/maxB token'].log( total_correct_maxB_tokens/total_tokens ) 
            args.neptune['logs/token'].log( total_correct_tokens/total_tokens ) 
            args.neptune['logs/rec_l2'].log( np.array(rec_l2).sum() ) 
            
        if model_wrapper.is_decoder():
            max_ids = -1
            for i in range(len(res_ids)):
                if len(res_ids[i]) > args.max_ids:
                    max_ids = args.max_ids
            predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores  = filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=max_ids)
            if len(predicted_sentences) < orig_batch['input_ids'].shape[0]:
                predicted_sentences += top_B_incorrect_sentences
                predicted_sentences_scores += top_B_incorrect_scores
        else:
            for l,token_type in sentence_ends:
                
                if args.l1_filter == 'maxB':
                    max_ids = args.batch_size
                elif args.l1_filter == 'all':
                    max_ids = -1
                else:
                    assert False

                if args.l2_filter == 'non-overlap':
                    correct_sentences = []
                    approx_sentences = []
                    approx_scores = []
                    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
                        if sc < args.l2_span_thresh:
                            correct_sentences.append( sent )
                        else:
                            approx_sentences.append( sent )
                            approx_scores.append( sc )

                    new_predicted_sentences, new_predicted_scores = filter(args, model_wrapper, R_Qs, l, token_type, res_ids, correct_sentences, approx_sentences, approx_scores, max_ids, args.batch_size)
                elif args.l2_filter == 'overlap':
                    new_predicted_sentences, new_predicted_scores = filter(args, model_wrapper, R_Qs, l, token_type, res_ids, [], [], [], max_ids, args.batch_size)
                else:
                    assert False

                predicted_sentences += new_predicted_sentences
                predicted_sentences_scores += new_predicted_scores

    correct_sentences = []
    approx_sentences = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    max_len = max( [len(s) for s in predicted_sentences] )
    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
        if sc < args.l2_span_thresh and args.defense_noise is None:
            correct_sentences.append( sent )
        else:
            approx_sentences.append( sent )
            approx_sentences_ext.append( sent + [-1]*(max_len - len(sent)) )
            approx_sentences_lens.append( len(sent) )
            approx_scores.append( sc )
    approx_scores = torch.tensor(approx_scores)
    approx_sentences_lens = torch.tensor(approx_sentences_lens)

    if len(approx_sentences) > 0:
        for i in range(len(correct_sentences)):
            sent = correct_sentences[i]
            similar_sentences = (torch.tensor(sent) == torch.tensor(approx_sentences_ext)[:,:len(sent)]).sum(1) >= torch.min(approx_sentences_lens,torch.tensor(len(sent)))*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf
        
        predicted_sentences = correct_sentences.copy()
        for i in range(len(correct_sentences), args.batch_size):
            idx = torch.argmin( approx_scores )
            predicted_sentences.append( approx_sentences[idx] )
            similar_sentences = (torch.tensor(approx_sentences_ext[idx]) == torch.tensor(approx_sentences_ext)).sum(1) >= max_len*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf

    for s in predicted_sentences:
        prediction.append( tokenizer.decode(s) )
    if args.neptune:
        args.neptune['logs/num_pred'].log( len(correct_sentences) ) 
    
    reference = []
    for i in range(orig_batch['input_ids'].shape[0]):
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i, :tokenizer.model_max_length], left=(args.pad=='left'))]
    if len(prediction) > len(reference):
        prediction = prediction[:len(reference)]

    if model_wrapper.is_decoder():
        new_prediction = []
        og_side = tokenizer.padding_side
        tokenizer.padding_side='right'
        for i in range(len(reference)):
            sequences = [reference[i]] + prediction
            batch = tokenizer(sequences,padding=True, truncation=True, return_tensors='pt')
            best_idx = (batch['input_ids'][1:] == batch['input_ids'][0]).sum(1).argmax()
            new_prediction.append(prediction[best_idx])
        tokenizer.padding_side=og_side
        prediction=new_prediction
    else:
        cost = np.zeros((len(prediction), len(prediction)))
        for i in range(len(prediction)):
            for j in range(len(prediction)):
                fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
                cost[i, j] = 1.0 - fm
        row_ind, col_ind = linear_sum_assignment(cost)

        ids = list(range(len(prediction)))
        ids.sort(key=lambda i: col_ind[i])
        new_prediction = []
        for i in range(len(prediction)):
            new_prediction += [prediction[ids[i]]]
        prediction = new_prediction

    return prediction, reference


def print_metrics(args, res, suffix):
    #sys.stderr.write(str(res) + '\n')
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        curr = res[metric].mid
        print(f'{metric:10} | fm: {curr.fmeasure*100:.3f} | p: {curr.precision*100:.3f} | r: {curr.recall*100:.3f}', flush=True)
        if args.neptune:
            args.neptune[f'logs/{metric}-fm_{suffix}'].log(curr.fmeasure*100)
            args.neptune[f'logs/{metric}-p_{suffix}'].log(curr.precision*100)
            args.neptune[f'logs/{metric}-r_{suffix}'].log(curr.recall*100)
    sum_12_fm = res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure
    if args.neptune:
        args.neptune[f'logs/r1fm+r2fm_{suffix}'].log(sum_12_fm*100)
    print(f'r1fm+r2fm = {sum_12_fm*100:.3f}', flush=True)
    print()

def main():
    device = torch.device(args.device)
    metric = load_metric('rouge', cache_dir=args.cache_dir)
    dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)

    model_wrapper = ModelWrapper(args)

    print('\n\nAttacking..\n', flush=True)
    predictions, references = [], []
    t_start = time.time()
    
    for i in range(args.start_input, min(args.n_inputs, args.end_input)):
        t_input_start = time.time()
        sample = dataset[i] # (seqs, labels)

        print(f'Running input #{i} of {args.n_inputs}.', flush=True)
        if args.neptune:
            args.neptune['logs/curr_input'].log(i)

        print('reference: ', flush=True)
        for seq in sample[0]:
            print('========================', flush=True)
            print(seq, flush=True)

        print('========================', flush=True)
        
        prediction, reference = reconstruct(args, sample, metric, model_wrapper)
        predictions += prediction
        references += reference

        print(f'Done with input #{i} of {args.n_inputs}.', flush=True)
        print('reference: ', flush=True)
        for seq in reference:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        print('predicted: ', flush=True)
        for seq in prediction:
            print('========================', flush=True)
            print(seq, flush=True)
        print('========================', flush=True)

        print('[Curr input metrics]:', flush=True)
        res = metric.compute(predictions=prediction, references=reference)
        print_metrics(args, res, suffix='curr')

        print('[Aggregate metrics]:', flush=True)
        res = metric.compute(predictions=predictions, references=references)
        print_metrics(args, res, suffix='agg')

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}', flush=True)
        print()
        print()

    print('Done with all.', flush=True)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)

if __name__ == '__main__':
    main()
