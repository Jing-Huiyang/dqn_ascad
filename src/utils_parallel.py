"""
并行化的攻击函数，用于加速模型选择过程
"""

import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
from functools import partial
from src.utils import AES_Sbox, AES_Sbox_inv, rk_key

def rank_compute_single_attack(args):
    """
    单次攻击的rank计算（用于并行化）
    """
    (ensemble_predictions, plt_attack, correct_key, leakage, dataset, byte, nb_traces, attack_idx, shuffle) = args
    
    hw = [bin(x).count("1") for x in range(256)]
    total_num_model = len(ensemble_predictions)
    
    # 检查空模型列表
    if total_num_model == 0:
        # 返回最差的rank evolution（所有值为255）
        return np.full(nb_traces, 255.0), np.zeros(256)
    
    # 准备数据
    ensemble_att_pred = []
    ensemble_att_plt = []
    
    for model_idx in range(total_num_model):
        if shuffle:
            # 为每次攻击设置不同的随机种子
            np.random.seed(attack_idx * 1000 + model_idx)
            indices = np.random.permutation(len(ensemble_predictions[model_idx]))[:nb_traces]
            att_pred = ensemble_predictions[model_idx][indices]
            att_plt = plt_attack[indices] if len(plt_attack.shape) == 1 else plt_attack
        else:
            att_pred = ensemble_predictions[model_idx][:nb_traces]
            att_plt = plt_attack[:nb_traces] if len(plt_attack.shape) == 1 else plt_attack
            
        ensemble_att_pred.append(att_pred)
        ensemble_att_plt.append(att_plt)
    
    # 计算rank evolution
    key_log_prob = np.zeros(256)
    (nb_traces_actual, nb_hyp) = ensemble_att_pred[0].shape
    rank_evol = np.full(nb_traces_actual, 255.0)
    
    for i in range(nb_traces_actual):
        for model_idx in range(total_num_model):
            prediction = ensemble_att_pred[model_idx]
            prediction_log = np.log(prediction + 1e-40)
            
            att_plt = ensemble_att_plt[model_idx]
            
            for k in range(256):
                if dataset == "AES_HD_ext":
                    if leakage == 'ID':
                        key_log_prob[k] += prediction_log[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11]]
                    else:
                        key_log_prob[k] += prediction_log[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11]]]
                elif dataset == "AES_HD_ext_ID":
                    if leakage == 'ID':
                        key_log_prob[k] += prediction_log[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])]]
                    else:
                        key_log_prob[k] += prediction_log[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])]]]
                elif dataset == "ASCON":
                    iv = [128, 64, 12, 6, 0, 0, 0, 0]
                    nonce0 = att_plt[i, 2]
                    nonce1 = att_plt[i, 3]
                    if leakage == 'ID':
                        Z = nonce0 ^ nonce1 ^ (iv[byte] & k) ^ (nonce1 & k) ^ k
                    elif leakage == 'HW':
                        Z = hw[nonce0 ^ nonce1 ^ (iv[byte] & k) ^ (nonce1 & k) ^ k]
                    key_log_prob[k] += prediction_log[i, Z]
                else:
                    # ASCAD等标准数据集
                    if leakage == 'ID':
                        key_log_prob[k] += prediction_log[i, AES_Sbox[k ^ int(att_plt[i])]]
                    else:
                        key_log_prob[k] += prediction_log[i, hw[AES_Sbox[k ^ int(att_plt[i])]]]
        
        rank_evol[i] = rk_key(key_log_prob, correct_key)
    
    return rank_evol, key_log_prob

def perform_attacks_ensemble_parallel(nb_traces, ensemble_predictions, plt_attack, correct_key, 
                                    leakage, dataset, nb_attacks=1, shuffle=True, byte=2, 
                                    n_processes=None):
    """
    并行化的集成攻击函数
    
    Args:
        nb_traces: 攻击使用的trace数量
        ensemble_predictions: 集成模型预测结果
        plt_attack: 攻击plaintext
        correct_key: 正确密钥
        leakage: 泄漏模型 ('HW' 或 'ID')
        dataset: 数据集名称
        nb_attacks: 攻击次数
        shuffle: 是否随机打乱数据
        byte: 目标字节
        n_processes: 并行进程数量（None为自动）
        
    Returns:
        avg_rank_evol: 平均rank演化
        key_log_prob: 最后一次攻击的密钥对数概率
    """
    
    if n_processes is None:
        n_processes = min(nb_attacks, mp.cpu_count())
    
    print(f"🚀 Running {nb_attacks} attacks in parallel using {n_processes} processes...")
    
    # 准备参数
    args_list = []
    for attack_idx in range(nb_attacks):
        args = (ensemble_predictions, plt_attack, correct_key, leakage, dataset, 
                byte, nb_traces, attack_idx, shuffle)
        args_list.append(args)
    
    # 并行执行
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    all_key_log_prob = np.zeros(256)
    
    if n_processes == 1:
        # 单进程执行（用于调试）
        print("Running in single process mode...")
        for i, args in enumerate(tqdm(args_list, desc="Processing attacks")):
            rank_evol, key_log_prob = rank_compute_single_attack(args)
            all_rk_evol[i] = rank_evol
            all_key_log_prob += key_log_prob
    else:
        # 多进程执行
        with mp.Pool(processes=n_processes) as pool:
            results = []
            
            # 提交所有任务
            for args in args_list:
                result = pool.apply_async(rank_compute_single_attack, (args,))
                results.append(result)
            
            # 收集结果
            for i, result in enumerate(tqdm(results, desc="Collecting results")):
                rank_evol, key_log_prob = result.get()
                all_rk_evol[i] = rank_evol
                all_key_log_prob += key_log_prob
    
    # 计算平均值
    avg_rank_evol = np.mean(all_rk_evol, axis=0)
    
    return avg_rank_evol, all_key_log_prob

def perform_attacks_parallel(nb_traces, predictions, plt_attack, correct_key, 
                           leakage, dataset, nb_attacks=1, shuffle=True, byte=2, 
                           n_processes=None):
    """
    并行化的单模型攻击函数
    """
    
    if n_processes is None:
        n_processes = min(nb_attacks, mp.cpu_count())
    
    # 将单模型转换为集成格式
    ensemble_predictions = [predictions]
    
    return perform_attacks_ensemble_parallel(
        nb_traces, ensemble_predictions, plt_attack, correct_key,
        leakage, dataset, nb_attacks, shuffle, byte, n_processes
    ) 