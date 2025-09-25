import torch
import numpy as np
import torch.nn.functional as F
from constants import config

def remove_padding(tokenizer, ids, left=False):
    if left:
        for i in range(ids.shape[0]):
            if ids[i].item() != config['PAD_TOKEN']:
                ids = ids[i:]
                break
    else:
        for i in range(ids.shape[0] - 1, -1, -1):
            if ids[i].item() != config['PAD_TOKEN']:
                ids = ids[:i+1]
                break
    return tokenizer.decode(ids)

def grad_dist(grads1, grads2, args):
    ret = 0.0
    n_g = 0
    for g1, g2 in zip(grads1, grads2):
        if (g1 is not None) and (g2 is not None):
            if args.loss == 'cos':
                ret += 1.0 - (g1 * g2).sum() / (g1.view(-1).norm(p=2) * g2.view(-1).norm(p=2) + 1e-9)
            elif args.loss == 'dlg':
                ret += (g1 - g2).square().sum()
            elif args.loss == 'tag':
                ret += (g1 - g2).square().sum() + args.tag_factor * torch.abs(g1 - g2).sum()
            else:
                assert False
            n_g += 1
    if args.loss == 'cos':
        ret /= n_g
    return ret


def get_closest_tokens(inputs_embeds, unused_tokens, embeddings_weight, metric='cos'):
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == 'l2':
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == 'cos':
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False

    d[:, :, unused_tokens] = 1e9
    return d, d.min(dim=2)[1]


def get_layer_decomp(grad, B=None, tol=None, upcast=False):
    grad = grad.detach().cpu().numpy()
    if upcast:
        grad = grad.astype(np.float32)
    if B == None:
        if upcast:
            B = np.linalg.matrix_rank( grad.astype(np.float32) , tol=tol)
            grad = grad.float()
        else:
            B = np.linalg.matrix_rank( grad , tol=tol)
    U,S,Vh = torch.svd_lowrank(torch.tensor(grad),q=B,niter=10)
    if upcast:
        R = Vh.T.half()
    else:
        R = Vh.T
    return  B, torch.Tensor(R).detach()

def get_perplexity(gpt2, x_embeds, bert_embeddings_weight, gpt2_embeddings_weight, c=0.1):
    gpt2_embeddings_weight = gpt2_embeddings_weight.repeat(x_embeds.shape[0], 1, 1)

    # Get alphas on BERT embeddings --> transfer to GPT-2
    alpha, _ = get_closest_tokens(x_embeds, bert_embeddings_weight)
    # alpha = torch.cdist(x_embeds[:, :-1, :], bert_embeddings_weight, p=2)
    alpha = F.softmax(-alpha/c, dim=2)
    gpt2_embeds = alpha.bmm(gpt2_embeddings_weight)

    # Pass through GPT-2 and get average perplexity
    out_gpt2 = gpt2(inputs_embeds=gpt2_embeds)
    log_probs = out_gpt2.logits.log_softmax(dim=2)
    fuzzy_perplexity = -(log_probs[:, :-1, :] * alpha[:, 1:, :]).sum(dim=2).mean(dim=1).sum()
    return fuzzy_perplexity


def check_if_in_span(R_K_norm, v, norm='l2'):
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v ) 
    out_of_span = proj - v
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)

    return size

def filter_in_span(R_K_norm, v, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)
    bools = size < thresh
    return torch.where( bools )

def get_top_B_in_span(R_K_norm, v, B, thresh, norm):
    size = check_if_in_span(R_K_norm, v, norm)   # 形状通常是 [N]
    bools = size < thresh                         # 过阈布尔掩码
    which = torch.where(bools)                    # (idx,)；若多维也适配
    _, idx = torch.sort(size[which])
    idx = idx[:B]
    which_new = [w[idx] for w in which]
    return tuple(which_new)                       # 一维时就是 (top_indices,)

def filter_outliers(d, stage='token', std_thrs=None, maxB=None):
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
    
def get_span_dists(args, model_wrapper, R_Qs, embeds, p=0, stage='token'):
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