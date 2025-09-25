import neptune.new as neptune 
import os
import numpy as np
import pandas as pd

gpt2_ids = [453, 452, 454, 455, 474, 471, 763, 768, #COLA
            457, 460, 459, 462, 475, 478, 762, 784, #SST2
            461, 465, 466, 467, 477, 476] #RT
#TODO: add B=64,128
llama_ids = [496, 498, 499, 516, 560, 518, (581, 657, 655, 656), # RT
             513, 512, 505, 508, 519, (559,), (599, 660), (661, 696, 697, 713), #SST2 - RUN B=32 FROM 79
             511, 506, 510, 504, 507, (598, 659), (715, 725, 723, 724)] #COLA - RUN B=32

enc_baselines_b8 = [791,  765, # COLA (TAG, COS, L1L2)
                    792,  766, #SST2 
                    790,  767]  #RT

# enc_baseline_ext = [349, 350, 352, 353, 356, 375, 400,
#                 354, 355, 358, 360, 403,
#                 359, 361, 365, 366, 367, 399, 
#                 363, 368, 369, 370, 374, 404]

enc_ids = [569,567,568, (605, ), #BERT SST2
           564, 653, (576, 658), (654, 646), #BERT RT
           572, 571, 570, 604] # BERT COLA

gpt2large_ids = [616, 617, 618, 620, 619, 621]
gpt2ft_ids = [640, 641, 642, 650, 651, 652]
gpt2nexttoken_ids = [634, 635, 636, 637, 638, 639]

dec_baselines = [685, 686, 687, 688, #TAG RT
                 693, 694, 698, 699, # TAG SST2
                 692, 691, 690, 689, # TAG COLA
                 (711, 726), (710, 722), (709, 721), (708, 720, 745), #LAMPCOS RT 
                 (703, 727), (702, 728), (701, 729), (700, 730), #LAMPCOS SST2 
                 (704, 716), (705, 717), (706, 718), (707, 719, 744), #LAMPCOS COLA
                 #LAMPL1L2 RT
                 734, 733, 732, 731, #LAML_L1L2 SST2
                 736, 737, (738, 746), (739, 747)] # LAMP_L1L2 COLA


ids = enc_ids + enc_baselines_b8 + gpt2_ids + llama_ids + gpt2large_ids + gpt2ft_ids + gpt2nexttoken_ids + dec_baselines
ids=[ f"LAM-{id}" if not isinstance(id, tuple) else tuple([f'LAM-{i}' for i in id]) for id in ids]
def get_param(project, param):
    if project[0].exists(param):
        return project[0][param].fetch()
    return np.nan

def get_stat_mean_std(projects, stat, starts):
    try:
        stats = []
        for i, project in enumerate(projects):
            stats.append(project[stat].fetch_values()['value'].to_numpy()[:(starts[i+1] - starts[i])])
        stats = np.concatenate(stats)
        stat_avg = stats.mean()
        stat_std = stats.std()
    except:
        return np.nan, np.nan
    
    return stat_avg, stat_std
    
df = pd.DataFrame(columns=['id', 'algorithm', 'model', 'batch_size', 'dataset', 
                             'rec_l1', 'rec_l1_err', 'rec_l2', 'rec_l2_err', 'rec_l1_maxb', 'rec_l1_maxb_err', 'maxb_frac', 'maxb_frac_err', 'token_frac', 'token_frac_err',
                             'rouge_1_fm', 'rouge_1_fm_err', 'rouge_1_p', 'rouge_1_p_err', 'rouge_1_r', 'rouge_1_r_err',
                             'rouge_2_fm', 'rouge_2_fm_err', 'rouge_2_p', 'rouge_2_p_err', 'rouge_2_r', 'rouge_2_r_err',
                             'rouge_l_fm', 'rouge_l_fm_err', 'rouge_l_p', 'rouge_l_p_err', 'rouge_l_r', 'rouge_l_r_err',
                             'rouge_lsum_fm', 'rouge_lsum_fm_err', 'rouge_lsum_p', 'rouge_lsum_p_err', 'rouge_lsum_r', 'rouge_lsum_r_err',
                             'elapsed_time'])
for id in ids:
    print(id, isinstance(id, tuple), type(id))
    if isinstance(id, tuple):
        project = []
        cutoffs = []
        for run_id in id:
            project.append(neptune.init_run(project="ethsri/LAMPlus", mode="read-only", with_id=f"{run_id}", api_token=os.environ['NEPTUNE_API_KEY']))
            cutoffs.append(get_param([project[-1]], 'parameters/start_input'))
        cutoffs.append(get_param([project[-1]], 'parameters/n_inputs'))
    else:
        project = [neptune.init_run(project="ethsri/LAMPlus", mode="read-only", with_id=f"{id}", api_token=os.environ['NEPTUNE_API_KEY'])]
        cutoffs = [0, 10000]
    batch_size = int(project[0]['parameters/batch_size'].fetch())
    dataset = get_param(project, 'parameters/dataset')
    model = get_param(project, 'parameters/model_path')
    try:
        algorithm = get_param(project, 'parameters/label').split('-')[0]
    except:
        algorithm = get_param(project, 'parameters/neptune_label').split('-')[:2]
        if algorithm[0] == 'lamp':
            algorithm = '-'.join(algorithm)
        else:
            algorithm = algorithm[0]
    rec_l1, rec_l1_err = get_stat_mean_std(project, 'logs/rec_l1', cutoffs)
    rec_l2, rec_l2_err = get_stat_mean_std(project, 'logs/rec_l2', cutoffs)
    rec_l1_maxb, rec_l1_maxb_err = get_stat_mean_std(project, 'logs/rec_l1_max_b', cutoffs)
    maxb_frac, maxb_frac_err = get_stat_mean_std(project, 'logs/maxB token', cutoffs)
    token_frac, token_frac_err = get_stat_mean_std(project, 'logs/token', cutoffs)
    
    rouge_1_fm, rouge_1_fm_err = get_stat_mean_std(project, 'logs/rouge1-fm_curr', cutoffs)
    rouge_1_p, rouge_1_p_err = get_stat_mean_std(project, 'logs/rouge1-p_curr', cutoffs)
    rouge_1_r, rouge_1_r_err = get_stat_mean_std(project, 'logs/rouge1-r_curr', cutoffs)
    
    rouge_2_fm, rouge_2_fm_err = get_stat_mean_std(project, 'logs/rouge2-fm_curr', cutoffs)
    rouge_2_p, rouge_2_p_err = get_stat_mean_std(project, 'logs/rouge2-p_curr', cutoffs)
    rouge_2_r, rouge_2_r_err = get_stat_mean_std(project, 'logs/rouge2-r_curr', cutoffs)
    
    rouge_l_fm, rouge_l_fm_err = get_stat_mean_std(project, 'logs/rougeL-fm_curr', cutoffs)
    rouge_l_p, rouge_l_p_err = get_stat_mean_std(project, 'logs/rougeL-p_curr', cutoffs)
    rouge_l_r, rouge_l_r_err = get_stat_mean_std(project, 'logs/rougeL-r_curr', cutoffs)
    
    rouge_lsum_fm, rouge_lsum_fm_err = get_stat_mean_std(project, 'logs/rougeLsum-fm_curr', cutoffs)
    rouge_lsum_p, rouge_lsum_p_err = get_stat_mean_std(project, 'logs/rougeLsum-p_curr', cutoffs)
    rouge_lsum_r, rouge_lsum_r_err = get_stat_mean_std(project, 'logs/rougeLsum-r_curr', cutoffs)
    
    elapsed_time = 0
    for proj in project:
        elapsed_time += proj['sys/running_time'].fetch()
    

    df.loc[len(df.index)] = [id, algorithm, model, batch_size, dataset, 
                             rec_l1, rec_l1_err, rec_l2, rec_l2_err, rec_l1_maxb, rec_l1_maxb_err, maxb_frac, maxb_frac_err, token_frac, token_frac_err,
                             rouge_1_fm, rouge_1_fm_err, rouge_1_p, rouge_1_p_err, rouge_1_r, rouge_1_r_err,
                             rouge_2_fm, rouge_2_fm_err, rouge_2_p, rouge_2_p_err, rouge_2_r, rouge_2_r_err,
                             rouge_l_fm, rouge_l_fm_err, rouge_l_p, rouge_l_p_err, rouge_l_r, rouge_l_r_err,
                             rouge_lsum_fm, rouge_lsum_fm_err, rouge_lsum_p, rouge_lsum_p_err, rouge_lsum_r, rouge_lsum_r_err,
                             elapsed_time]

df = df.set_index('id')
df.to_csv('results.csv')
    
    
