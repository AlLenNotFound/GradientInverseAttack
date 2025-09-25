
import copy
import torch
from utils.functional import check_if_in_span, get_span_dists, filter_outliers
import itertools
from tqdm import tqdm
import numpy as np
def filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=-1):
    def repetition_penalty(batch, window=8, penalty=0.8):
        if batch.size(1) <= 1:
            return torch.ones(batch.size(0), device=batch.device, dtype=torch.float32)
        w = min(window, batch.size(1))
        ctx = batch[:, -w:]
        mult = torch.ones(batch.size(0), device=batch.device, dtype=torch.float32)
        for i in range(ctx.size(0)):
            _, counts = torch.unique(ctx[i], return_counts=True)
            m = counts.max().item()
            if m > 1:
                mult[i] = mult[i] * (penalty ** (m - 1))
        return mult

    # --- 小工具：把 sizesq2 -> 序列分数（0.7*最后一步 + 0.3*均值） ---  # >>> CHANGED
    def seq_score_from_sizes(sizesq2, batch_for_penalty):
        if model_wrapper.has_bos():
            seq_mean = sizesq2[:, 1:].mean(dim=1)
        else:
            seq_mean = sizesq2.mean(dim=1)
        last = sizesq2[:, -1]
        seq_score = 0.7 * last + 0.3 * seq_mean

        # 重复惩罚
        rp = repetition_penalty(
            batch_for_penalty,
            window=getattr(args, 'repeat_window', 8),
            penalty=getattr(args, 'repeat_penalty', 0.8),
        )
        seq_score = seq_score / rp  # 惩罚 -> 分数变大，更难被选
        # —— 立即重复硬惩罚（新增）——
        if batch_for_penalty.size(1) >= 2:
            repeat2 = (batch_for_penalty[:, -1] == batch_for_penalty[:, -2])
            if repeat2.any():
                seq_score[repeat2] = seq_score[repeat2] + float(getattr(args, "repeat_hard_pen", 0.20))  # >>> ADD

        # 语言模型复排（可选）
        lam = float(getattr(args, 'lm_rerank_lambda', 0.3))
        if lam > 0 and hasattr(model_wrapper, 'forward_logits'):
            with torch.no_grad():
                attn = (batch_for_penalty != model_wrapper.pad_token).long()
                logits = model_wrapper.forward_logits(batch_for_penalty, attention_mask=attn)
                last_logits = logits[:, -1, :]
                last_token = batch_for_penalty[:, -1]
                nll = torch.nn.functional.cross_entropy(last_logits, last_token, reduction='none')
            seq_score = (1 - lam) * seq_score + lam * nll
        return seq_score

    R_Q2 = R_Qs[1]
    res_ids = copy.deepcopy(res_ids)
    # 限制每个位置最多max_ids个candidate
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
            ends = torch.tensor(res_ids[0], dtype=torch.long)
            ends = ends[ends != model_wrapper.pad_token]
        else:
            ends = torch.tensor(res_ids[i], dtype=torch.long)
        if ends.numel() == 0:
            break

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
            ends_per = max(ends.shape[0], 1)
            par_chunk = max(args.parallel // ends_per, 1)
            for _ in range(par_chunk * ends_per):
                el = next(it_lst, None)
                if el is None:
                    break
                els_b.append(el[0])
                els_ends.append(el[1])
            els_b = torch.tensor(np.array(els_b), dtype=torch.long)
            els_ends = torch.tensor(np.array(els_ends), dtype=torch.long)
            if els_b.shape[0] == 0 and is_complete:
                break
            elif els_b.shape[0] == 0:
                idxs = np.array(list(itertools.product(range(batch.shape[0]), range(len(ends)))))
                new_batch = torch.cat((torch.tensor(batch[idxs[:, 0]]).long(),\
                                       torch.tensor(ends[idxs[:, 1]]).long().unsqueeze(1)), dim=-1).to(args.device)
                is_new_batch_incorrect = is_batch_incorrect[idxs[:, 0]].to(args.device)
                sizesq2 = torch.cat(ds)
                sizesq2, correct_sentences = filter_outliers(sizesq2, stage='sequence', std_thrs=args.l2_std_thrs, maxB=args.batch_size)
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

            seq_score_all = seq_score_from_sizes(sizesq2, new_batch)
            next_batch.append(new_batch[correct_sentences].to('cpu'))
            next_scores.append(seq_score_all[correct_sentences].to('cpu'))
            is_next_batch_incorrect.append(is_new_batch_incorrect[correct_sentences].to('cpu'))

            curr_sentence += len(els_b) // max(ends.shape[0], 1)

            # 处理“不通过”的分支：也用相同的 seq_score 逻辑  # >>> CHANGED
            incorrect_mask = ~correct_sentences
            incorrect_sentences = new_batch[incorrect_mask]
            inc_scores = seq_score_all[incorrect_mask]  # 已包含重复惩罚 + (可选)LM

            if incorrect_sentences.size(0) > 0:
                scores_best_batch, sentences_best_batch = [], []
                inc_scores_clone = inc_scores.clone()
                for _ in range(args.batch_size):
                    idx_best_batch = torch.argmin(inc_scores_clone)
                    best_score = inc_scores_clone[idx_best_batch]
                    best_sentence = incorrect_sentences[idx_best_batch]
                    sentences_best_batch.append(best_sentence.cpu().numpy().tolist())
                    scores_best_batch.append(best_score.item())

                    # 去重
                    similar_sentences = (best_sentence == incorrect_sentences).sum(1) >= (i + 1) * args.distinct_thresh
                    inc_scores_clone[similar_sentences] = torch.inf

                # 插回 top_B_incorrect_*   # 保持你原逻辑
                for b_idx in range(len(scores_best_batch)):
                    if scores_best_batch[b_idx] > top_B_incorrect_scores_len[-1]:
                        break
                    predicted_idx = 0
                    while scores_best_batch[b_idx] > top_B_incorrect_scores_len[predicted_idx]:
                        predicted_idx += 1
                    for rep_idx in range(predicted_idx, args.batch_size):
                        if len(top_B_incorrect_sentences_len[rep_idx]) > 0 and \
                                (torch.tensor(sentences_best_batch[b_idx:b_idx + 1]) == torch.tensor(
                                    top_B_incorrect_sentences_len[rep_idx:rep_idx + 1])).sum(1) \
                                >= (i + 1) * args.distinct_thresh:
                            continue
                        else:
                            top_B_incorrect_sentences_len = (
                                    top_B_incorrect_sentences_len[:predicted_idx]
                                    + sentences_best_batch[b_idx:b_idx + 1]
                                    + top_B_incorrect_sentences_len[predicted_idx:rep_idx]
                                    + top_B_incorrect_sentences_len[rep_idx + 1:]
                            )
                            top_B_incorrect_scores_len = (
                                    top_B_incorrect_scores_len[:predicted_idx]
                                    + scores_best_batch[b_idx:b_idx + 1]
                                    + top_B_incorrect_scores_len[predicted_idx:rep_idx]
                                    + top_B_incorrect_scores_len[rep_idx + 1:]
                            )
                            break

            # if use_bar:
            progress_bar.update(len(els_b))

            # 汇总到下一轮
        # batch = torch.cat(next_batch) if next_batch else torch.empty((0, 1), dtype=torch.long)

            # next_batch.append(new_batch[correct_sentences].to('cpu'))
            # if model_wrapper.has_bos():
            #     next_scores.append(sizesq2[:, 1:].mean(dim=1)[correct_sentences].to('cpu'))
            # else:
            #     next_scores.append(sizesq2.mean(dim=1)[correct_sentences].to('cpu'))
            # is_next_batch_incorrect.append(is_new_batch_incorrect[correct_sentences].to('cpu'))
            #
            # curr_sentence += len(els_b)//ends.shape[0]
            #
            # incorrect_sentences = new_batch[~correct_sentences]
            # if model_wrapper.has_bos():
            #     sizesq2_incorrect = sizesq2[~correct_sentences, 1:].mean(dim=1)
            # else:
            #     sizesq2_incorrect = sizesq2[~correct_sentences].mean(dim=1)
            #
            # if len(incorrect_sentences) == 0:
            #     continue
            #
            # # Draw unique
            # scores_best_batch, sentences_best_batch = [], []
            # for b_idx in range(args.batch_size):
            #     idx_best_batch = torch.argmin(sizesq2_incorrect)
            #     best_score = sizesq2_incorrect[idx_best_batch]
            #     best_sentence = incorrect_sentences[idx_best_batch]
            #     sentences_best_batch.append( best_sentence.cpu().numpy().tolist() )
            #     scores_best_batch.append( best_score.item() )
            #     similar_sentences = (best_sentence == incorrect_sentences).sum(1) >= (i+1)*args.distinct_thresh
            #     sizesq2_incorrect[similar_sentences] = torch.inf
            #
            # for b_idx in range(len(scores_best_batch)):
            #     if scores_best_batch[b_idx] > top_B_incorrect_scores_len[-1]:
            #         break
            #     predicted_idx = 0
            #     while scores_best_batch[b_idx] > top_B_incorrect_scores_len[predicted_idx]:
            #         predicted_idx += 1
            #     for rep_idx in range(predicted_idx, args.batch_size):
            #         if len(top_B_incorrect_sentences_len[rep_idx]) > 0 and\
            #             (torch.tensor(sentences_best_batch[b_idx:b_idx+1]) == torch.tensor(top_B_incorrect_sentences_len[rep_idx:rep_idx+1])).sum(1) \
            #             >= (i+1)*args.distinct_thresh:
            #
            #             continue
            #         else:
            #             top_B_incorrect_sentences_len = top_B_incorrect_sentences_len[:predicted_idx] + sentences_best_batch[b_idx:b_idx+1] + top_B_incorrect_sentences_len[predicted_idx:rep_idx] +top_B_incorrect_sentences_len[rep_idx+1:]
            #             top_B_incorrect_scores_len = top_B_incorrect_scores_len[:predicted_idx] + scores_best_batch[b_idx:b_idx+1] + top_B_incorrect_scores_len[predicted_idx:rep_idx] + top_B_incorrect_scores_len[rep_idx+1:]
            #             break
            # progress_bar = tqdm(total=max(batch.shape[0] * max(ends.shape[0], 1), 1))
        if len(next_batch) == 0:
            break

        batch = torch.cat(next_batch)
        if len(batch) == 0:
            break
        is_batch_incorrect = torch.cat(is_next_batch_incorrect) if len(is_next_batch_incorrect) > 0 else torch.empty(
            (0,), dtype=torch.long)
        scores = torch.cat(next_scores) if len(next_scores) > 0 else torch.empty((0,), dtype=torch.float32)
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
    # for i in range(batch.shape[0]):
    #     predicted_sentences.append(batch[i].cpu().numpy().tolist())
    #     predicted_sentences_scores.append(scores[i].item())
    # Add remaining sentences if any
    if 'batch' in locals() and isinstance(batch, torch.Tensor) and batch.shape[0] > 0:
        for i in range(batch.shape[0]):
            predicted_sentences.append(batch[i].cpu().numpy().tolist())
            predicted_sentences_scores.append(scores[i].item())

    # 兜底：如果仍然没有任何候选，用错误候选或种子 token 构一个最短序列
    if len(predicted_sentences) == 0:
        if any(len(s) > 0 for s in top_B_incorrect_sentences):
            # 选分数最小的错误候选
            idx = int(np.argmin([sc for sc, s in zip(top_B_incorrect_scores, top_B_incorrect_sentences) if len(s) > 0]))
            predicted_sentences = [top_B_incorrect_sentences[idx]]
            predicted_sentences_scores = [top_B_incorrect_scores[idx]]
        else:
            # 实在没法子：用首位 token 做 1-token 序列
            seed = res_ids[0][0] if len(res_ids) > 0 and len(res_ids[0]) > 0 else model_wrapper.eos_token
            predicted_sentences = [[int(seed)]]
            predicted_sentences_scores = [float('inf')]

    return predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores

# def filter_decoder_step(args, model_wrapper, R_Qs, batch, p):
#     if args.defense_noise is None:
#         R_Q2 = R_Qs[1]
#         attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
#         input_layer1 = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask)[0]
#         sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
#         boolsq2 = sizesq2 < args.l2_span_thresh      # [B, T]，True=过阈
#         # -------- 新增：用“通过比例”而不是“所有位置都必须过阈” --------
#         keep_ratio = getattr(args, "l2_keep_ratio", 0.95)  # 没加到命令行也能用
#         if model_wrapper.has_rope():
#             mask_tokens = [model_wrapper.pad_token]
#             if model_wrapper.start_token is not None:
#                 mask_tokens.append(model_wrapper.start_token)
#             mask_tokens = torch.tensor(mask_tokens, device=batch.device)
#             # 掩码位 & BOS 位置直接视为通过
#             boolsq2 = torch.logical_or(boolsq2, torch.isin(batch, mask_tokens))
#             pass_ratio = boolsq2.float().mean(dim=1)          # 每条序列过阈比例
#             correct_sentences = pass_ratio >= keep_ratio
#             # 保留你原来的“重复序列”过滤
#             if p > 1 and (model_wrapper.start_token is not None):
#                 repeats = (batch[:, -2] == model_wrapper.start_token) & \
#                          (torch.isin(batch[:, -1], batch[:, 1].to(batch.device)))
#                 correct_sentences = correct_sentences & (~repeats)
#         else:
#             pass_ratio = boolsq2.float().mean(dim=1)
#             correct_sentences = pass_ratio >= keep_ratio
#         return sizesq2, correct_sentences
#
#     else:
#         attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
#         input_layers = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask, layers=args.n_layers-1)
#         return get_span_dists(args, model_wrapper, R_Qs, input_layers, stage='sequence')
def filter_decoder_step(args, model_wrapper, R_Qs, batch, p):
    if args.defense_noise is None:
        R_Q2 = R_Qs[1]
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layer1 = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask)[0]
        sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)

        # --- 自适应阈值（可选） ---  # >>> CHANGED
        l2_th = float(args.l2_span_thresh)
        if bool(getattr(args, 'l2_span_auto', False)):
            step_d = sizesq2[:, -1]
            q = float(getattr(args, 'l2_auto_quantile', 0.25))
            l2_th = min(l2_th, torch.quantile(step_d, q).item())

        boolsq2 = sizesq2 < l2_th  # True=过阈

        # --- 判定逻辑：优先用“最近 k 位全过阈”，否则用“通过比例” ---  # >>> CHANGED
        k = int(getattr(args, 'require_last_k', 0))
        if model_wrapper.has_rope():
            mask_tokens = [model_wrapper.pad_token]
            if model_wrapper.start_token is not None:
                mask_tokens.append(model_wrapper.start_token)
            mask_tokens = torch.tensor(mask_tokens, device=batch.device)
            boolsq2 = torch.logical_or(boolsq2, torch.isin(batch, mask_tokens))

        if k > 0 and k <= boolsq2.shape[1]:
            correct_sentences = boolsq2[:, -k:].all(dim=1)
        else:
            keep_ratio = float(getattr(args, "l2_keep_ratio", 0.95))
            pass_ratio = boolsq2.float().mean(dim=1)
            correct_sentences = pass_ratio >= keep_ratio

        # 你的重复过滤逻辑保留
        if model_wrapper.has_rope() and p > 1 and (model_wrapper.start_token is not None):
            repeats = (batch[:, -2] == model_wrapper.start_token) & \
                      (torch.isin(batch[:, -1], batch[:, 1].to(batch.device)))
            correct_sentences = correct_sentences & (~repeats)

        return sizesq2, correct_sentences

    else:
        attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
        input_layers = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask, layers=args.n_layers-1)
        return get_span_dists(args, model_wrapper, R_Qs, input_layers, stage='sequence')
