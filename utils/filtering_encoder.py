import torch
import itertools
import numpy as np
from tqdm import tqdm
from utils.functional import check_if_in_span

def filter_encoder(args, model_wrapper, R_Q2, l, token_type, res_ids, sentence_filter, approx_sentence_filter, approx_sentence_score, max_ids, B):
    predicted_sentences = [ [-1]*(l+1) for i in range(B) ]
    predicted_sentences_scores = [ torch.inf for i in range(B) ]

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
