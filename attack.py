import datetime
import numpy as np
import torch
import torch.nn.functional as F
from evaluate import load as load_metric
from utils.models import ModelWrapper
from utils.data import TextDataset
from utils.filtering_encoder import filter_encoder
from utils.filtering_decoder import filter_decoder
from utils.functional import get_top_B_in_span, check_if_in_span, remove_padding, filter_outliers, get_span_dists
from args_factory import get_args
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# old seed: 100
args = get_args()
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0
@torch.no_grad()
def check_position_recall(res_ids_s, gold_ids, pad_id=None, eos_id=None, max_print=10, tag=""):
    """
    res_ids_s: List[List[token_id]]，稀疏性筛选后每个位置的候选池
    gold_ids : List[int]，该样本的金标序列（已去 padding）
    返回：dict（召回统计），并打印前若干个 miss 的具体位置/ID
    """
    misses = []
    n_pos = min(len(res_ids_s), len(gold_ids))
    for p in range(n_pos):
        g = int(gold_ids[p])
        if pad_id is not None and g == pad_id:
            break
        if eos_id is not None and g == eos_id:
            # 若你的约定是到 EOS 停止拼接，可在此 break；否则保留
            pass
        cand_set = set(res_ids_s[p])
        if g not in cand_set:
            misses.append((p, g))
    total = len(gold_ids)
    found = total - len(misses)
    print(f"[Recall@Pos{tag}] total={total} | miss={len(misses)} | recall={found/total:.4f}")
    if misses:
        print(f"[Recall@Pos{tag}] first_misses: {misses[:max_print]}")
    return {
        "total": total,
        "miss": len(misses),
        "recall": found / total,
        "miss_list": misses,
    }


def get_mandatory_punct_ids(tokenizer):
    punct_strs = [
        ".", ",", "!", "?", ";", ":", "...", "—", "-", "(", ")", "[", "]", "{", "}",
        "'", "\"", "’", "”", "“", "…", "\n", "\r", "\t", " ", "—", "–"
    ]
    ids = set()
    for s in punct_strs:
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            for t in toks:
                ids.add(int(t))
        except Exception:
            pass
    for s in ["\n\n", "\r\n", "\n \n"]:
        try:
            toks = tokenizer.encode(s, add_special_tokens=False)
            for t in toks:
                ids.add(int(t))
        except Exception:
            pass
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if eos_id is not None and eos_id != -1:
        ids.add(int(eos_id))
    if pad_id is not None and pad_id != -1:
        ids.add(int(pad_id))
    return ids


# @torch.no_grad()
# def sparsify_res_ids_per_position(
#     args, model_wrapper, res_ids_s, grad_slices_per_head,
#     eps=None, tau=None, K_keep=None, keep_head=None, always_topm=None, min_keep=None
# ):
#     """
#     用 Hessian/Gauss-Newton 曲率替代原稀疏率做位置级剪枝：
#       对候选 e：C_h(e) = || e @ G_h ||^2，跨头取“最小的 k 个 head”的均值 C_small(e)。
#       小 C_small 更好（更平/不敏感），在 C_small 上做阈值或 Top-k 筛。
#     其它流程（always_topm/min_keep/K_keep/mandatory/暖启）不变。
#     """
#     if grad_slices_per_head is None or len(res_ids_s) == 0:
#         return res_ids_s
#
#     # 超参（兼容你原来的命名）
#     K_keep      = K_keep      if K_keep      is not None else getattr(args, "pre_sparse_K_keep", 48)
#     keep_head   = keep_head   if keep_head   is not None else getattr(args, "pre_sparse_keep_head", -1)  # -1=全部
#     always_topm = always_topm if always_topm is not None else getattr(args, "pre_sparse_always_topm", 16)
#     min_keep    = min_keep    if min_keep    is not None else getattr(args, "pre_sparse_min_keep", 24)
#
#     # Hessian/GN 相关
#     head_topk_div   = getattr(args, "hess_head_topk_div", 5)   # 取最小的 H/5 个 head 融合
#     curv_use_quant  = getattr(args, "hess_use_quantile", True) # True: 分位阈；False: 比例 Top-k
#     curv_q          = getattr(args, "hess_q", 0.50)            # 分位阈（小曲率保留）
#     curv_keep_ratio = getattr(args, "hess_keep_ratio", 0.50)   # 备用：比例 Top-k
#     per_dim_norm    = getattr(args, "hess_per_dim_norm", True) # 对 g 的各维做标准化
#     head_warmup     = getattr(args, "pre_sparse_head_warmup", 8)
#     tail_warmup     = getattr(args, "pre_sparse_tail_warmup", 5)
#
#     H = len(grad_slices_per_head)
#     heads = range(H) if keep_head == -1 else range(min(keep_head, H))
#
#     res_filtered = []
#     m = model_wrapper.model
#     dev = model_wrapper.device
#
#     for p, ids_p in enumerate(res_ids_s):
#         K = len(ids_p)
#         if K == 0:
#             res_filtered.append(ids_p)
#             continue
#
#         ids_t = torch.as_tensor(ids_p, device=dev, dtype=torch.long)
#
#         # === 把 E 对齐到“第1层 Q 线性层的输入空间” ===
#         try:
#             if model_wrapper.is_decoder() and hasattr(m, "transformer") and hasattr(m.transformer.h[0], "ln_1"):
#                 # GPT-2 系：E = ln_1(wte[token] + wpe[pos])
#                 wte = m.transformer.wte.weight            # [V,d]
#                 wpe = m.transformer.wpe.weight[p]         # [d]
#                 x   = wte.index_select(0, ids_t) + wpe.unsqueeze(0)
#                 E   = m.transformer.h[0].ln_1(x)
#
#             elif hasattr(getattr(m, "model", None), "layers") and hasattr(m.model.layers[0], "input_layernorm"):
#                 # LLaMA 系：E = input_layernorm(embed_tokens[token])
#                 wte = m.model.embed_tokens.weight
#                 x   = wte.index_select(0, ids_t)
#                 E   = m.model.layers[0].input_layernorm(x)
#
#             elif hasattr(m, "embeddings") and hasattr(m.embeddings, "LayerNorm"):
#                 # BERT 系：E = LayerNorm(we[token] + pe[pos] + te[0])
#                 we  = m.embeddings.word_embeddings.weight
#                 pe  = m.embeddings.position_embeddings.weight[p]
#                 te  = m.embeddings.token_type_embeddings.weight[0]
#                 x   = we.index_select(0, ids_t) + pe.unsqueeze(0) + te.unsqueeze(0)
#                 E   = m.embeddings.LayerNorm(x)
#
#             else:
#                 # 兜底：退回旧路径（可能不完全对齐）
#                 full_emb_pos = model_wrapper.get_embeddings(p)[0]  # [V,d]
#                 E = full_emb_pos.index_select(0, ids_t)
#
#         except Exception:
#             # 任意异常则退回旧路径
#             full_emb_pos = model_wrapper.get_embeddings(p)[0]
#             E = full_emb_pos.index_select(0, ids_t)
#
#         E = E.to(dev).float()  # [K,d]
#
#         # =============== 1) 计算曲率 C_all：各 head 的 ||E @ G_h||^2 ===============
#         curvs = []
#         for hi in heads:
#             Gh = grad_slices_per_head[hi].to(dev).float()        # [d, d_head]
#             g  = E @ Gh                                          # [K, d_head]
#             if per_dim_norm:
#                 g = g / (g.std(dim=0, keepdim=True) + 1e-8)      # 维度标准化，稳定量纲
#             c_h = (g * g).sum(dim=1)                             # [K]
#             curvs.append(c_h)
#         C_all = torch.stack(curvs, dim=1)                         # [K, H_used]
#
#         # 跨头融合：取“最小的 k_head 个 head”的均值（小曲率更好）
#         k_head = max(1, int(C_all.shape[1] / max(1, head_topk_div)))
#         C_small = torch.topk(C_all, k=k_head, dim=1, largest=False).values.mean(dim=1)  # [K]
#         print(f"[pre-sparse/Hess][pos={p}] C.min={C_small.min().item():.4f}  "
#               f"C.mean={C_small.mean().item():.4f}  C.max={C_small.max().item():.4f}  K={K}")
#
#         # =============== 2) 自适应保底/上限（按 K 比例） ===============
#         always_topm_eff = max(always_topm, int(0.125 * K))
#         min_keep_eff    = max(min_keep,    int(0.35  * K))
#         K_keep_eff      = max(K_keep,      int(0.50  * K))
#
#         # 句首/尾 warmup：跳过剪枝
#         if p < head_warmup or p > (tail_warmup + 1e9):
#             res_filtered.append(ids_p)
#             continue
#
#         # 必保留：原始前 top-m + 标点/特殊 token
#         keep = set(range(min(always_topm_eff, K)))
#         mandatory_ids = getattr(model_wrapper, "_mandatory_punct_ids", None)
#         if mandatory_ids is None:
#             mandatory_ids = get_mandatory_punct_ids(model_wrapper.tokenizer)
#             setattr(model_wrapper, "_mandatory_punct_ids", mandatory_ids)
#         keep |= {i for i, tok in enumerate(ids_p) if tok in mandatory_ids}
#
#         # =============== 3) 基于曲率的筛选（小曲率保留） ===============
#         if curv_use_quant:
#             thr  = torch.quantile(C_small, q=curv_q)
#             good = torch.where(C_small <= thr)[0].tolist()
#         else:
#             kk   = max(1, int(curv_keep_ratio * K))
#             good = torch.topk(-C_small, k=min(kk, K), largest=True).indices.tolist()  # 对 -C_small 取 topk
#         keep.update(good)
#
#         # 不足补齐 → 多了截断（优先小曲率）
#         if len(keep) < min_keep_eff:
#             rest = [i for i in range(K) if i not in keep]
#             if len(rest) > 0:
#                 add = torch.topk(-C_small[rest],
#                                  k=min(min_keep_eff - len(keep), len(rest)),
#                                  largest=True).indices.tolist()
#                 keep.update([rest[i] for i in add])
#
#         keep_list = sorted(list(keep))
#         if len(keep_list) > K_keep_eff:
#             pri = list(range(min(always_topm_eff, K))) + [i for i, t in enumerate(ids_p) if t in mandatory_ids]
#             pri = list(dict.fromkeys(pri))  # 去重
#             others = [i for i in keep_list if i not in pri]
#             top_others = []
#             if len(others) > 0:
#                 top_others = torch.topk(-C_small[others],
#                                         k=min(K_keep_eff - len(pri), len(others)),
#                                         largest=True).indices.tolist()
#                 top_others = [others[i] for i in top_others]
#             keep_list = (pri + top_others)[:K_keep_eff]
#
#         ids_new = [ids_p[i] for i in keep_list]
#         res_filtered.append(ids_new)
#
#         print(f"[pre-sparse/Hess] pos={p} K:{len(ids_p)} -> {len(ids_new)} (kept)")
#
#     return res_filtered


@torch.no_grad()
def sparsify_res_ids_per_position(
    args, model_wrapper, res_ids_s, grad_slices_per_head,
    eps=None, tau=None, K_keep=None, keep_head=None, always_topm=None, min_keep=None
):
    """
    对单个样本的 res_ids_s（List[List[token_id]]] 做位置级稀疏筛选。
    - 稀疏分数：对每个候选 token v，在位置 p 取 e(v,p) 与每个 head 的 G_h 相乘：
        g_h = e(v,p) @ G_h -> [d_head]，稀疏率 = mean( |g_h| <= eps )
      跨 head 取平均，得到 S(v,p) ∈ [0,1]
    - 保守兜底：
        1) 至少保留 always_topm 个原始 Top（按你 Stage2 的原顺序）
        2) 稀疏率 >= tau 的全部保留
        3) 若仍不足 min_keep，再补齐到 min_keep（按稀疏率从高到低）
        4) 最终最多保留 K_keep
    """
    if grad_slices_per_head is None or len(res_ids_s) == 0:
        return res_ids_s

    # 超参（可从 args 里带默认值）
    eps = eps if eps is not None else getattr(args, "pre_sparse_eps", 1e-3)
    tau = tau if tau is not None else getattr(args, "pre_sparse_tau", 0.35)
    K_keep = K_keep if K_keep is not None else getattr(args, "pre_sparse_K_keep", 48)
    keep_head = keep_head if keep_head is not None else getattr(args, "pre_sparse_keep_head", -1)  # -1=用全部head
    always_topm = always_topm if always_topm is not None else getattr(args, "pre_sparse_always_topm", 16)
    min_keep = min_keep if min_keep is not None else getattr(args, "pre_sparse_min_keep", 24)

    H = len(grad_slices_per_head)
    heads = range(H) if keep_head == -1 else range(min(keep_head, H))

    res_filtered = []
    for p, ids_p in enumerate(res_ids_s):
        K = len(ids_p)
        ids_t = torch.as_tensor(ids_p, device=model_wrapper.device, dtype=torch.long)
        full_emb_pos = model_wrapper.get_embeddings(p)[0]  # [V, d_model]
        E = full_emb_pos.index_select(0, ids_t)  # [K, d_model]

        scores = []
        for hi in heads:
            G = grad_slices_per_head[hi]  # [d_model, d_head]
            g = E @ G  # [K, d_head]
            thr = torch.quantile(g.abs(), q=getattr(args, "pre_sparse_q", 0.25), dim=0, keepdim=True)  # [1, d_head]
            s_h = (g.abs() <= thr).float().mean(dim=1)
            scores.append(s_h)

        # 融合（默认 median）
        # fuse = getattr(args, "pre_sparse_fuse", "median")
        S_all = torch.stack(scores, dim=1)
        k_head = max(1, int(S_all.shape[1] / 5))  # H/3 可调
        S = torch.topk(S_all, k=k_head, dim=1, largest=True).values.mean(dim=1)
        # —— 融合（默认 median）之后 —— S: [K]
        print(f"[pre-sparse][pos={p}] S.min={S.min().item():.4f}  "
              f"S.mean={S.mean().item():.4f}  S.max={S.max().item():.4f}  K={len(ids_p)}")

        # —— 自适应保底/上限（按K）——
        always_topm_eff = max(always_topm, int(0.125 * K))
        min_keep_eff = max(min_keep, int(0.35 * K))
        K_keep_eff = max(K_keep, int(0.5 * K))

        # 位置例外：句首/末尾跳过或放宽
        if p < getattr(args, "pre_sparse_head_warmup", 8) or \
                p > (getattr(args, "pre_sparse_tail_warmup", 5) + 1e9):  # 你可以改为基于 eff_lenEach[b]
            res_filtered.append(ids_p)  # 跳过筛选
            continue
        # 必保留：原始前 top-m + 标点/特殊token
        keep = set(range(min(always_topm_eff, K)))
        mandatory_ids = getattr(model_wrapper, "_mandatory_punct_ids", None)
        if mandatory_ids is None:
            mandatory_ids = get_mandatory_punct_ids(model_wrapper.tokenizer)
            setattr(model_wrapper, "_mandatory_punct_ids", mandatory_ids)
        keep |= {i for i, tok in enumerate(ids_p) if tok in mandatory_ids}

        # 稀疏率阈值（分位数版本建议 tau=0.55~0.65）
        tau_eff = getattr(args, "pre_sparse_tau", 0.6)
        good = torch.where(S >= tau_eff)[0].tolist()
        keep.update(good)

        # 保底到 min_keep_eff
        if len(keep) < min_keep_eff:
            rest = [i for i in range(K) if i not in keep]
            add = torch.topk(S[rest], k=min(min_keep_eff - len(keep), len(rest)), largest=True).indices.tolist()
            keep.update([rest[i] for i in add])

        # 最多保留 K_keep_eff（优先保留 always_topm 与标点，然后按 S 截断）
        keep_list = sorted(list(keep))
        if len(keep_list) > K_keep_eff:
            pri = list(range(min(always_topm_eff, K))) + [i for i, t in enumerate(ids_p) if t in mandatory_ids]
            pri = list(dict.fromkeys(pri))  # 去重
            others = [i for i in keep_list if i not in pri]
            top_others = []
            if len(others) > 0:
                top_others = torch.topk(S[others], k=min(K_keep_eff - len(pri), len(others)),
                                        largest=True).indices.tolist()
                top_others = [others[i] for i in top_others]
            keep_list = (pri + top_others)[:K_keep_eff]

        ids_new = [ids_p[i] for i in keep_list]
        res_filtered.append(ids_new)

        # 可选：打印剪枝比例
        print(f"[pre-sparse] pos={p} K:{len(ids_p)} -> {len(ids_new)} (kept)")

    return res_filtered

def to_text_list(x, tokenizer):
    # 统一把各种形态转成 List[str]
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            return []
        # 已经是字符串列表
        if isinstance(x[0], str):
            return list(x)
        # 是一整句的 id 扁平列表：[int, int, ...]
        if isinstance(x[0], int):
            return [tokenizer.decode(list(x), skip_special_tokens=True)]
        # 是多个句子的 id 列表：[[...], [...], ...]
        out = []
        for t in x:
            if isinstance(t, str):
                out.append(t)
            elif hasattr(t, "tolist"):  # torch.Tensor / np.ndarray
                out.append(tokenizer.decode(list(t.tolist()), skip_special_tokens=True))
            elif isinstance(t, (list, tuple)):
                out.append(tokenizer.decode(list(t), skip_special_tokens=True))
            else:
                out.append(str(t))
        return out
    # torch.Tensor / np.ndarray / 其他
    if hasattr(x, "tolist"):
        arr = x.tolist()
        if isinstance(arr, list) and (len(arr) > 0) and isinstance(arr[0], int):
            return [tokenizer.decode(arr, skip_special_tokens=True)]
        return [str(arr)]
    return [str(x)]


def _recall_report(ref_ids, pool_ids, res_ids):
    pool = set(pool_ids.tolist() if hasattr(pool_ids, "tolist") else pool_ids)
    miss1, miss2 = [], []
    for p, t in enumerate(ref_ids):
        if t not in pool:
            miss1.append((p, int(t)))
        elif p < len(res_ids) and int(t) not in set(res_ids[p]):
            miss2.append((p, int(t)))
    print(f"[Recall] len={len(ref_ids)} | Stage1-miss={len(miss1)} | Stage2-miss={len(miss2)}")
    if miss1: print("[Recall][S1] first misses:", miss1[:10])
    if miss2: print("[Recall][S2] first misses:", miss2[:10])

import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

# def beam_search_decoder(args, model_wrapper, R_Qs, res_ids, forced_start_token=None):
#     # beam_width = 3 if args.max_ids <= 0 else args.max_ids
#     beam_width = 3
#     R_Q2 = R_Qs[1]
#     device = model_wrapper.device
#
#     # 只解一段合理的长度（见下一节），避免 512 全部跑一遍
#     max_steps = getattr(args, "beam_max_steps", None)
#     if max_steps is not None:
#         res_iter = res_ids[:max_steps]
#     else:
#         res_iter = res_ids
#
#     beams = [([], 0.0)]
#
#     from tqdm import tqdm
#     pbar = tqdm(total=len(res_iter), desc="BeamSearch positions", dynamic_ncols=True)
#
#     for pos_idx, pos_candidates in enumerate(res_iter):
#         if len(beams) == 0 or len(pos_candidates) == 0:
#             break
#         # if pos_idx == 0 and forced_start_token is not None:
#         #     pos_candidates = [forced_start_token]
#         # else:
#         #     pos_candidates = list(pos_candidates)
#
#         # ----- 批量构造所有新分支 -----
#         B, K = len(beams), len(pos_candidates)
#         # 前缀复制 B 次，每个前缀接上 K 个候选
#         expanded = []
#         prev_scores = []
#         for seq, score in beams:
#             for tok in pos_candidates:
#                 expanded.append(seq + [tok])
#                 prev_scores.append(score)
#
#         new_tensor = torch.tensor(expanded, device=device)          # [B*K, pos+1]
#         with torch.no_grad():
#             hs = model_wrapper.get_layer_inputs(new_tensor)[0]      # [B*K, pos+1, d]
#             last = hs[:, -1, :]                                     # [B*K, d]
#             dist = check_if_in_span(R_Q2, last, args.dist_norm)     # [B*K]  距离小更好
#
#         prev_scores = torch.tensor(prev_scores, device=device)      # [B*K]
#         total_scores = prev_scores + dist                           # 你的原始打分
#
#         # ----- 剪枝：选最小的 beam_width 个 -----
#         k = min(beam_width, total_scores.numel())
#         top_idx = torch.topk(-total_scores, k=k, largest=True).indices  # 取负号=选最小
#         beams = [(expanded[i], total_scores[i].item()) for i in top_idx.tolist()]
#
#         # 早停：如果 EOS 被选中且 beams 全部以 EOS 结尾，可停止
#         eos_id = getattr(model_wrapper.tokenizer, "eos_token_id", None)
#         if eos_id is not None and all(len(b[0]) > 0 and b[0][-1] == eos_id for b in beams):
#             pbar.update(1)
#             pbar.set_postfix_str(f"pos={pos_idx} |B|={len(beams)} |EOS stop")
#             break
#
#         pbar.update(1)
#         pbar.set_postfix_str(f"pos={pos_idx} |B|={len(beams)}")
#
#     pbar.close()
#
#     final_sentences = [seq for seq, score in beams]
#     final_scores = [score for seq, score in beams]
#
#     # 返回两个列表：句子列表和对应的分数列表
#     return final_sentences, final_scores
def beam_search_decoder(args, model_wrapper, R_Qs, res_ids, forced_start_token=None):
    beam_width = 3
    R_Q2 = R_Qs[1]
    device = model_wrapper.device

    # 只解一段合理的长度，避免全长都跑
    max_steps = getattr(args, "beam_max_steps", None)
    res_iter = res_ids[:max_steps] if max_steps is not None else res_ids

    # beams: (seq_ids_list, total_score)；只存“累计后的总分”
    beams = [([], 0.0)]

    from tqdm import tqdm
    pbar = tqdm(total=len(res_iter), desc="BeamSearch positions", dynamic_ncols=True)

    # 确保 eval（dropout 关闭），和 L2 校验一致
    if hasattr(model_wrapper, "model"):
        model_wrapper.model.eval()
    torch.set_grad_enabled(False)

    for pos_idx, pos_candidates in enumerate(res_iter):
        if len(beams) == 0:
            break
        if len(pos_candidates) == 0:
            break

        # 强制起始 token（锚定第一步，更稳）
        if pos_idx == 0 and (forced_start_token is not None):
            if forced_start_token in pos_candidates:
                pos_candidates = [forced_start_token]
            else:
                pos_candidates = [forced_start_token] + list(pos_candidates)

        B, K = len(beams), len(pos_candidates)

        # -------- 向量化构造 expanded（避免 Python 双循环） --------
        base = torch.tensor([b[0] for b in beams], device=device, dtype=torch.long)  # [B, pos]
        cand = torch.tensor(pos_candidates, device=device, dtype=torch.long)         # [K]

        # 组合出 [B*K, pos+1]
        base_rep = base.repeat_interleave(K, dim=0)               # [B*K, pos]
        cand_rep = cand.repeat(B).view(-1, 1)                     # [B*K, 1]
        new_tensor = torch.cat([base_rep, cand_rep], dim=1)       # [B*K, pos+1]

        # -------- 与 L2 校验一致地取 last（注意 mask/bert 分支） --------
        with torch.no_grad():
            if model_wrapper.is_decoder():
                attn = torch.ones_like(new_tensor, dtype=torch.long)
                hs = model_wrapper.get_layer_inputs(new_tensor, attention_mask=attn)[0]
            elif model_wrapper.is_bert():
                token_type_ids = torch.zeros_like(new_tensor, dtype=torch.long)
                hs = model_wrapper.get_layer_inputs(new_tensor, token_type_ids=token_type_ids)[0]
            else:
                hs = model_wrapper.get_layer_inputs(new_tensor)[0]

            last = hs[:, -1, :]                                   # [B*K, d]
            dist = check_if_in_span(R_Q2, last, args.dist_norm)    # [B*K]  距离小更好

        # 只累计一次总分（prev_total 广播）
        prev_total = torch.tensor([b[1] for b in beams], device=device).repeat_interleave(K)  # [B*K]
        new_total = prev_total + dist                                                         # [B*K]

        # -------- 选最小的 beam_width 个 --------
        k = min(beam_width, new_total.numel())
        top_idx = torch.topk(-new_total, k=k, largest=True).indices  # 取负号=选“更小的距离”

        # 回写 beams（把 new_tensor 的行转回 list[int]）
        new_seqs  = new_tensor[top_idx].tolist()
        new_scores = new_total[top_idx].tolist()
        beams = list(zip(new_seqs, new_scores))

        # 早停：全 EOS 则停
        eos_id = getattr(model_wrapper.tokenizer, "eos_token_id", None)
        if eos_id is not None and all(len(b[0]) > 0 and b[0][-1] == eos_id for b in beams):
            pbar.update(1)
            pbar.set_postfix_str(f"pos={pos_idx} |B|={len(beams)} |EOS stop")
            break

        pbar.update(1)
        pbar.set_postfix_str(f"pos={pos_idx} |B|={len(beams)}")

    pbar.close()

    final_sentences = [seq for seq, score in beams]
    final_scores = [score for seq, score in beams]
    return final_sentences, final_scores

def reconstruct_headwise_pooled(args, model_wrapper, true_grads, R_Qs_original, head_R_Qs_split, input_batch):
    tokenizer = model_wrapper.tokenizer

    # ----------(NEW) 先算有效长度与 ref_ids，Stage-1 也要用 ----------
    if "attention_mask" in input_batch:
        eff_len_each = input_batch["attention_mask"].sum(dim=1)  # [B]
        eff_len = int(eff_len_each.max().item())
    else:
        ids = input_batch["input_ids"]  # [B, T]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            eff_len_each = torch.full((ids.size(0),), ids.size(1), device=ids.device)
            eff_len = int(ids.size(1))
        else:
            eff_len_each = (ids != pad_id).sum(dim=1)  # [B]
            eff_len = int(eff_len_each.max().item())
    Bsz = input_batch["input_ids"].size(0)
    ref_ids_list = []
    for b in range(Bsz):
        Lb = int(eff_len_each[b].item())
        ref_ids_list.append(input_batch["input_ids"][b].tolist()[:Lb])
    # if "attention_mask" in input_batch:
    #     eff_len = int(input_batch["attention_mask"][0].sum().item())
    # else:
    #     ids0 = input_batch["input_ids"][0]
    #     pad_id = tokenizer.pad_token_id
    #     eff_len = int((ids0 != pad_id).sum().item()) if pad_id is not None else int(ids0.shape[0])
    # ref_ids = input_batch["input_ids"][0].tolist()[:eff_len]

    # =================================================================
    # 阶段一: 基于多头的全局粗筛（替换为“头间融合 + 自适应加码”）
    # =================================================================
    print("Stage 1: Generating global candidate pool using head-wise projection...")

    word_embeddings = model_wrapper.get_word_embeddings().to(args.device).float()  # [V, d_model]

    # 只取 Query(Q) 部分
    grad_l1 = true_grads[model_wrapper.layer_ids[0]]            # e.g. [768, 2304] (GPT-2)
    d_model = model_wrapper.model.config.hidden_size            # 768
    grad_l1_query = grad_l1[:, :d_model].to(args.device).float()# [768, 768]

    # 准备每个 head 的 Q 子矩阵（与 R_Q_i 空间对齐）
    h = model_wrapper.model.config.num_attention_heads
    d_head = d_model // h
    grad_slices_per_head = [
        grad_l1_query[:, i * d_head : (i + 1) * d_head]         # [768, 64]
        for i in range(h)
    ]

    def build_pool_unsupervised(R_Q_heads, grad_slices, wte, args, tokenizer):
        # 1) 各 head 的到子空间残差 [H, V]
        head_scores = []
        for R_Q_i, G_i in zip(R_Q_heads, grad_slices):
            proj = torch.matmul(wte, G_i)  # [V, d_head]
            dist = check_if_in_span(R_Q_i, proj, args.dist_norm).float()  # [V]
            head_scores.append(dist.unsqueeze(0))
        scores = torch.cat(head_scores, dim=0)  # [H, V]
        fuse_min = scores.min(dim=0).values  # [V]
        fuse_mean = scores.mean(dim=0)  # [V]

        def pool_for_k(k):
            idx_min = torch.topk(fuse_min, k=k, largest=False).indices
            idx_mean = torch.topk(fuse_mean, k=max(1, k // 2), largest=False).indices
            pool = torch.unique(torch.cat([idx_min, idx_mean]))
            if getattr(args, 'always_include_eos_period', True):
                extras = torch.tensor([13, tokenizer.eos_token_id or 50256], device=wte.device)
                pool = torch.unique(torch.cat([pool, extras]))
            return pool

        target = getattr(args, "target_pool", 3500)  # 建议先放大一点，见下方参数建议
        k_lo, k_hi = 64, getattr(args, "k_per_head_max", 4096)
        best = pool_for_k(k_hi)
        while k_lo <= k_hi:
            mid = (k_lo + k_hi) // 2
            pool = pool_for_k(mid)
            if pool.numel() >= target:
                best = pool
                k_hi = mid - 1
            else:
                k_lo = mid + 1

        if best.numel() > target:
            best = best[:target]
        return best, fuse_min

    candidate_pool_indices, fuse_min = build_pool_unsupervised(
        head_R_Qs_split, grad_slices_per_head, word_embeddings, args, tokenizer
    )

    # （可选但建议开启，调试期“强保证”：把所有 ref token 并入 pool）
    # need = torch.tensor(sorted(set(ref_ids)), device=candidate_pool_indices.device)
    # candidate_pool_indices = torch.unique(torch.cat([candidate_pool_indices, need]), sorted=True)
    # ---------- Stage-1 位置感知 booster（无监督，不看 reference） ----------
    # 覆盖句首若干位置 + 末位（句号），避免只看 0,1,2
    P_front = min(getattr(args, "booster_front_positions", 5), eff_len)  # 覆盖前 5 个位置
    P = list(range(P_front))
    if eff_len > 0 and (eff_len - 1) not in P:
        P.append(eff_len - 1)
    m_each = getattr(args, "booster_topm", 800)  # 每个位置取 top-m（建议略放宽）

    boost_ids = set()
    for p in P:
        full_emb_pos = model_wrapper.get_embeddings(p)[0]  # [V, d]
        dists_p = check_if_in_span(R_Qs_original[0], full_emb_pos, args.dist_norm)  # [V]
        topm = torch.topk(dists_p, k=min(m_each, dists_p.numel()), largest=False).indices
        boost_ids.update(topm.tolist())

    if boost_ids:
        device = args.device
        boost_ids = torch.tensor(sorted(boost_ids), device=device)

        # 并集
        union = torch.unique(torch.cat([candidate_pool_indices, boost_ids]))

        # --------- 关键：mandatory = booster ∪ {'.', EOS}，裁剪时必须保留 ----------
        mandatory = set(boost_ids.tolist())
        if getattr(args, 'always_include_eos_period', True):
            mandatory.update([13, tokenizer.eos_token_id or 50256])
        mandatory = torch.tensor(sorted(mandatory), device=device)

        # 只在 union 太大时裁剪；裁剪 = mandatory 全保留 + 用 fuse_min 从剩余里补足
        target = getattr(args, "target_pool", 3500)
        if union.numel() > target:
            # 剩余可选集 = union \ mandatory
            mask = torch.ones(union.numel(), dtype=torch.bool, device=device)
            # 把 mandatory 在 union 中的位置打掉
            if mandatory.numel() > 0:
                # 建一个哈希表再比对，避免 O(N^2)
                mand_set = set(mandatory.tolist())
                for i, t in enumerate(union.tolist()):
                    if t in mand_set:
                        mask[i] = False
            residual = union[mask]  # 还可以被打分竞争的集合

            need = max(0, target - mandatory.numel())
            if residual.numel() > 0 and need > 0:
                residual_scores = fuse_min[residual]  # 用 head-wise 的 fused 分数排序
                keep_idx = torch.topk(residual_scores, k=min(need, residual.numel()), largest=False).indices
                candidate_pool_indices = torch.cat([mandatory, residual[keep_idx]])
            else:
                # mandatory 已经超过 target（极端情况），那就截断 mandatory
                candidate_pool_indices = mandatory[:target]
        else:
            candidate_pool_indices = union
    # ---------------------------------------------------------------------

    print(f"Stage 1 finished. Candidate pool size: {candidate_pool_indices.numel()}")

    # =================================================================
    # 阶段二: 位置级筛选（你原来的逻辑保留）
    # =================================================================
    print("Stage 2: Position-specific filtering within the pool (per-sample)...")
    R_Q_full = R_Qs_original[0]
    eos_id = getattr(tokenizer, "eos_token_id", None)


    # ---------- CHANGE #3: 按样本循环，逐位置筛 ----------
    per_sample_outputs = []  # List[ (res_pos_i, res_ids_i, res_types_i, sentence_ends_i) ]

    for b in range(Bsz):
        Lb = int(eff_len_each[b].item())
        Lb = min(Lb, args.max_len, tokenizer.model_max_length)

        res_pos_i, res_ids_i, res_types_i = [], [], []
        sentence_ends_i = []

        for p in range(Lb):
            if candidate_pool_indices.numel() == 0:
                print(f"[b={b}] Candidate pool is empty, stopping.")
                break

            # full_emb_pos = full_emb_pos_cache[p]                          # [V, d]
            # candidate_pos_embeds = full_emb_pos[candidate_pool_indices]   # [V_pool, d]
            with torch.no_grad():
                full_emb_pos = model_wrapper.get_embeddings(p)[0]  # [V, d]
                candidate_pos_embeds = full_emb_pos.index_select(0, candidate_pool_indices)  # [V_pool, d]
            del full_emb_pos

            K2 = getattr(args, "pos_topk", 128)

            if model_wrapper.is_bert():
                _, top_indices_in_pool, types_in_pool = get_top_B_in_span(
                    R_Q_full, candidate_pos_embeds, K2, args.l1_span_thresh, args.dist_norm
                )
            else:
                top_indices_in_pool, = get_top_B_in_span(
                    R_Q_full, candidate_pos_embeds, K2, args.l1_span_thresh, args.dist_norm
                )
                types_in_pool = torch.zeros_like(top_indices_in_pool)

            cur_count = int(top_indices_in_pool.numel())
            print(f"[Stage2][b={b}] pos={p} | actual_candidates={cur_count}")

            final_token_ids_for_p = candidate_pool_indices[top_indices_in_pool]
            ids_p = final_token_ids_for_p.tolist()
            types_p = types_in_pool.tolist()

            # EOS 截断：该样本自己的句尾
            if eos_id is not None and eos_id in ids_p:
                end_token_ind = ids_p.index(eos_id)
                sentence_token_type = types_p[end_token_ind]
                sentence_ends_i.append((p, sentence_token_type))
                ids_p = ids_p[:end_token_ind]
                types_p = types_p[:end_token_ind]

            if args.max_ids > 0:
                ids_p = ids_p[:args.max_ids]
                types_p = types_p[:args.max_ids]

            if len(ids_p) == 0:
                continue

            res_ids_i.append(ids_p)
            res_types_i.append(types_p)
            res_pos_i += [p] * len(ids_p)

            # 逐样本召回报告
            _recall_report(ref_ids_list[b], candidate_pool_indices, res_ids_i)

        per_sample_outputs.append((res_pos_i, res_ids_i, res_types_i, sentence_ends_i))

    # ---------- CHANGE #4: 返回“每个样本一份”的列表 ----------
    return per_sample_outputs


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
                d = get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(d, std_thrs=std_thrs, maxB=max(50*model_wrapper.args.batch_size, int(0.05*len(model_wrapper.tokenizer))))
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

def reconstruct(args, device, sample, metric, model_wrapper: ModelWrapper):
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
        # B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        B, R_Qs_original, head_R_Qs_split = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        R_Q = R_Qs_original[0]
        R_Q2 = R_Qs_original[1]

        # —— 放在 reconstruct(...) 里，B, R_Qs_original, head_R_Qs_split 之后 ——
        grad_l1 = true_grads[model_wrapper.layer_ids[0]]  # [d_model, 3*d_model] 类似
        d_model = model_wrapper.model.config.hidden_size
        grad_l1_query = grad_l1[:, :d_model].to(args.device).float()  # 只取 Q 部分 [d_model, d_model]
        h = model_wrapper.model.config.num_attention_heads
        d_head = d_model // h
        grad_slices_per_head = [grad_l1_query[:, i * d_head:(i + 1) * d_head] for i in
                                range(h)]  # [H]个 [d_model, d_head]

        if B is None:
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]
            return ['' for _ in range(len(reference))], reference
        R_Q, R_Q2 = R_Q.to(args.device), R_Q2.to(args.device)
        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range( orig_batch['input_ids'].shape[1] ):
            total_true_token_count2 += args.batch_size - ( orig_batch['input_ids'][:,i] == model_wrapper.pad_token).sum()
            uniques = torch.unique(orig_batch['input_ids'][:,i])
            total_true_token_count += uniques.numel()
            if model_wrapper.pad_token in uniques.tolist():
                total_true_token_count -= 1

        print(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        if args.neptune:
            args.neptune['logs/max_rank'].log( B )
            args.neptune['logs/batch_tokens'].log( total_true_token_count2 )
            args.neptune['logs/batch_unique_tokens'].log( total_true_token_count )

        if args.headwise_factorization:
            per_samples = reconstruct_headwise_pooled(args, model_wrapper, true_grads,
                                                                                     R_Qs_original, head_R_Qs_split, orig_batch)
        else:
            res_pos, res_ids, res_types, sentence_ends = filter_l1(args, model_wrapper, R_Qs_original)

        if args.max_ids > 0:
            per_samples = [
                (rp, [c[:args.max_ids] for c in ri], rt, se)
                for (rp, ri, rt, se) in per_samples
            ]
        del true_grads

        all_empty = all(len(ri) == 0 for (_, ri, _, _) in per_samples)
        if all_empty:
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad == 'left'))]
            return ['' for _ in reference], reference
        rec_l1, rec_l1_maxB, rec_l2 = [], [], []

        Bsz = orig_batch['input_ids'].shape[0]
        for s in range(Bsz):
            _, res_ids_s, _, sentence_ends_s = per_samples[s]

            sentence_in = True
            sentence_in_max_B = True
            orig_sentence = orig_batch['input_ids'][s]
            last_idx = torch.where(orig_batch['input_ids'][s] != tokenizer.pad_token_id)[0][-1].item()

            for pos, token in enumerate(orig_sentence):
                if not model_wrapper.is_bert() and pos == last_idx:
                    break
                if not model_wrapper.has_rope():
                    if pos >= len(res_ids_s):
                        sentence_in = False
                        break

                if token == model_wrapper.pad_token and args.pad == 'right':
                    pos -= 1
                    break
                elif token == model_wrapper.pad_token and args.pad == 'left':
                    continue

                if model_wrapper.has_rope():
                    in_set = token in res_ids_s[0]
                    in_set_maxB = token in res_ids_s[0][:min(args.batch_size, len(res_ids_s[0]))]
                else:
                    in_set = token in res_ids_s[pos]
                    in_set_maxB = token in res_ids_s[pos][:min(args.batch_size, len(res_ids_s[pos]))]

                total_correct_tokens += 1 if in_set else 0
                total_correct_maxB_tokens += 1 if in_set_maxB else 0
                total_tokens += 1

                if token == model_wrapper.eos_token and args.pad == 'right':
                    break

                sentence_in = sentence_in and in_set
                sentence_in_max_B = sentence_in_max_B and in_set_maxB

            if model_wrapper.is_bert():
                sentence_in = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends_s
                sentence_in_max_B = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends_s

            rec_l1.append(sentence_in)
            rec_l1_maxB.append(sentence_in_max_B)

            # L2（保持你原逻辑）
            if model_wrapper.is_bert():
                token_type_ids = (orig_batch['token_type_ids'][s][:orig_sentence.shape[0]]).reshape(1, -1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence[:orig_sentence.shape[0]].reshape(1, -1),
                                                              token_type_ids)[0]
            else:
                attention_mask = orig_batch['attention_mask'][s][:min(last_idx + 1, orig_sentence.shape[0])].reshape(1,
                                                                                                                     -1)
                input_layer1 = \
                model_wrapper.get_layer_inputs(orig_sentence[:min(last_idx + 1, orig_sentence.shape[0])].reshape(1, -1),
                                               attention_mask=attention_mask)[0]

            sizesq2 = check_if_in_span(R_Q2, input_layer1, args.dist_norm)
            boolsq2 = sizesq2 < args.l2_span_thresh
            if args.task == 'next_token_pred':
                rec_l2.append(torch.all(boolsq2[:-1]).item())
            elif model_wrapper.has_rope():
                rec_l2.append(torch.all(boolsq2[1:]).item())
            else:
                rec_l2.append(torch.all(boolsq2).item())

        print( f'Rec L1: {rec_l1}, Rec L1 MaxB: {rec_l1_maxB}, Rec MaxB Token: {total_correct_maxB_tokens/total_tokens}, Rec Token: {total_correct_tokens/total_tokens}, Rec L2: {rec_l2}' )

        if args.neptune:
            args.neptune['logs/rec_l1'].log( np.array(rec_l1).sum() )
            args.neptune['logs/rec_l1_max_b'].log( np.array(rec_l1_maxB).sum() )
            args.neptune['logs/maxB token'].log( total_correct_maxB_tokens/total_tokens )
            args.neptune['logs/token'].log( total_correct_tokens/total_tokens )
            args.neptune['logs/rec_l2'].log( np.array(rec_l2).sum() )

        if model_wrapper.is_decoder():
            predicted_sentences, predicted_sentences_scores = [], []
            for s in range(Bsz):
                _, res_ids_s, _, _ = per_samples[s]

                res_ids_s = sparsify_res_ids_per_position(
                    args, model_wrapper, res_ids_s, grad_slices_per_head,
                    eps=getattr(args, "pre_sparse_eps", 1e-3),
                    tau=getattr(args, "pre_sparse_tau", 0.35),
                    K_keep=getattr(args, "pre_sparse_K_keep", 32),
                    keep_head=getattr(args, "pre_sparse_keep_head", -1),  # -1=用所有head
                    always_topm=getattr(args, "pre_sparse_always_topm", 16),  # 始终保留的位置Top-m
                    min_keep=getattr(args, "pre_sparse_min_keep", 24)  # 保底保留
                )
                # 逐位确认真 token 是否仍在位置池子里

                ids_full = orig_batch['input_ids'][s]  # [T]
                if "attention_mask" in orig_batch:
                    L_eff = int(orig_batch["attention_mask"][s].sum().item())
                    gold_ids = ids_full[:L_eff].tolist()
                else:
                    pad_id = tokenizer.pad_token_id
                    if pad_id is None:
                        gold_ids = ids_full.tolist()
                    else:
                        gold_ids = ids_full[ids_full != pad_id].tolist()

                pad_id = tokenizer.pad_token_id
                eos_id = getattr(tokenizer, "eos_token_id", None)
                _ = check_position_recall(
                    res_ids_s,
                    gold_ids,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    tag=f" b={s}"
                )

                start_tok = int(orig_batch['input_ids'][s, 0].item())
                # print(res_ids_s)

                seqs_s, scores_s = beam_search_decoder(
                    args, model_wrapper, R_Qs_original, res_ids_s, forced_start_token=start_tok
                )

                if len(seqs_s) == 0:
                    # 兜底：L1贪心
                    greedy = [c[0] for c in res_ids_s if len(c) > 0]
                    if len(greedy) > 0:
                        seqs_s, scores_s = [greedy], [0.0]
                if len(seqs_s) > 0:
                    best = int(np.argmin(scores_s))
                    predicted_sentences.append(seqs_s[best])
                    predicted_sentences_scores.append(scores_s[best])
                else:
                    predicted_sentences.append([])
                    predicted_sentences_scores.append(1e9)

            prediction_ids = predicted_sentences
        else:
            # 每个样本独立用自己的 sentence_ends_s / res_ids_s 跑 filter_encoder
            predicted_sentences, predicted_sentences_scores = [], []
            if args.l1_filter == 'maxB':
                max_ids = args.batch_size
            elif args.l1_filter == 'all':
                max_ids = -1
            else:
                assert False
            Bsz = orig_batch['input_ids'].shape[0]
            for s in range(Bsz):
                _, res_ids_s, _, sentence_ends_s = per_samples[s]
                if args.l2_filter == 'non-overlap':
                    # 这里没有“来自别处的”先验候选，所以作为空集传入即可
                    correct_sentences = []
                    approx_sentences = []
                    approx_scores = []
                    for (l, token_type) in sentence_ends_s:
                        new_pred_sents, new_pred_scores = filter_encoder(
                            args, model_wrapper, R_Q2, l, token_type,
                            res_ids_s,
                            correct_sentences, approx_sentences, approx_scores,
                            max_ids, args.batch_size
                        )
                        predicted_sentences += new_pred_sents
                        predicted_sentences_scores += new_pred_scores
                elif args.l2_filter == 'overlap':
                    for (l, token_type) in sentence_ends_s:
                        new_pred_sents, new_pred_scores = filter_encoder(
                            args, model_wrapper, R_Q2, l, token_type,
                            res_ids_s,
                            [], [], [],
                            max_ids, args.batch_size
                        )
                        predicted_sentences += new_pred_sents
                        predicted_sentences_scores += new_pred_scores
                elif args.l2_filter == 'all':
                    # 如果你的 filter_encoder 的 'all' 分支需要 anchor_set，可以按该样本长度构造
                    eff_len_i = int(orig_batch["attention_mask"][s].sum().item()) \
                        if "attention_mask" in orig_batch else int(
                        (orig_batch["input_ids"][s] != tokenizer.pad_token_id).sum().item())
                    anchor_set = set(range(eff_len_i))
                    # 如果后面根本没用到 anchor_set，可以不传；否则依据你的 filter_encoder 实现来调用
                    for (l, token_type) in sentence_ends_s:
                        new_pred_sents, new_pred_scores = filter_encoder(
                            args, model_wrapper, R_Q2, l, token_type,
                            res_ids_s,
                            [], [], [],
                            max_ids, args.batch_size
                        )
                        predicted_sentences += new_pred_sents
                        predicted_sentences_scores += new_pred_scores
                else:
                    assert False

                # predicted_sentences += new_predicted_sentences
                # predicted_sentences_scores += new_predicted_scores

                prediction_ids = []
                for s in range(Bsz):
                    _, res_ids_s, _, _ = per_samples[s]
                    greedy = [c[0] for c in res_ids_s if len(c) > 0]
                    prediction_ids.append(greedy)

    print("=========================================\n",predicted_sentences,"=========================================\n")

    # 防守式兜底（极少见）：若某个样本空序列，用该样本的 L1 贪心
    for s in range(len(prediction_ids)):
        if not prediction_ids[s]:
            _, res_ids_s, _, _ = per_samples[s]
            greedy = [c[0] for c in res_ids_s if len(c) > 0]
            prediction_ids[s] = greedy if len(greedy) > 0 else []

    # 转成文本
    prediction = to_text_list(prediction_ids, model_wrapper.tokenizer)

    reference = []
    for i in range(orig_batch['input_ids'].shape[0]):
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i, :tokenizer.model_max_length],
                                     left=(args.pad == 'left'))]
    reference = to_text_list(reference, model_wrapper.tokenizer)
    return prediction, reference


def print_metrics(args, res, suffix):
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        fm = res[metric] * 100
        print(f'{metric:10} | fm: {fm:.3f}', flush=True)
        if args.neptune:
            args.neptune[f'logs/{metric}-fm_{suffix}'].log(fm)
    sum_12_fm = (res['rouge1'] + res['rouge2']) * 100
    print(f'r1fm+r2fm = {sum_12_fm:.3f}', flush=True)
    if args.neptune:
        args.neptune[f'logs/r1fm+r2fm_{suffix}'].log(sum_12_fm)



def main():
    device = torch.device(args.device)
    metric = load_metric('rouge', cache_dir=args.cache_dir)
    print("Creating TextDataset with:", args.dataset, args.split)
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
        
        prediction, reference = reconstruct(args, device, sample, metric, model_wrapper)
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

        tok = model_wrapper.tokenizer
        prediction_texts = to_text_list(prediction, tok)
        reference_texts = to_text_list(reference, tok)

        print('[Curr input metrics]:', flush=True)
        res = metric.compute(predictions=prediction_texts, references=reference_texts)
        print_metrics(args, res, suffix='curr')

        input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
        total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
        print(f'input #{i} time: {input_time} | total time: {total_time}', flush=True)
        print()
        print()

    print('Done with all.', flush=True)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)

if __name__ == '__main__':
    print("Dataset argument:", args.dataset)
    main()
