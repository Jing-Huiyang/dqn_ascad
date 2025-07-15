"""
优化的并行攻击函数，解决数据传输瓶颈
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from tqdm import tqdm
import random
from functools import partial
from src.utils import AES_Sbox, rk_key

# 全局变量用于存储共享数据
_shared_predictions = None
_shared_plt_attack = None
_shared_metadata = None

def init_worker(shared_predictions_name, shared_plt_name, shape_info, metadata):
    """
    工作进程初始化函数，设置共享内存
    """
    global _shared_predictions, _shared_plt_attack, _shared_metadata
    
    # 连接到共享内存
    shm_predictions = shared_memory.SharedMemory(name=shared_predictions_name)
    shm_plt = shared_memory.SharedMemory(name=shared_plt_name)
    
    # 重建numpy数组
    predictions_shape, plt_shape = shape_info
    _shared_predictions = np.ndarray(predictions_shape, dtype=np.float32, buffer=shm_predictions.buf)
    _shared_plt_attack = np.ndarray(plt_shape, dtype=np.float32, buffer=shm_plt.buf)
    _shared_metadata = metadata

def rank_compute_optimized(attack_idx):
    """
    优化的单次攻击计算（使用共享内存）
    """
    global _shared_predictions, _shared_plt_attack, _shared_metadata
    
    # 解包元数据
    correct_key, leakage, dataset, byte, nb_traces, shuffle = _shared_metadata
    
    hw = [bin(x).count("1") for x in range(256)]
    total_num_model = len(_shared_predictions)
    
    # 准备数据
    ensemble_att_pred = []
    ensemble_att_plt = []
    
    for model_idx in range(total_num_model):
        if shuffle:
            np.random.seed(attack_idx * 1000 + model_idx)
            indices = np.random.permutation(len(_shared_predictions[model_idx]))[:nb_traces]
            att_pred = _shared_predictions[model_idx][indices]
            att_plt = _shared_plt_attack[indices] if len(_shared_plt_attack.shape) == 1 else _shared_plt_attack
        else:
            att_pred = _shared_predictions[model_idx][:nb_traces]
            att_plt = _shared_plt_attack[:nb_traces] if len(_shared_plt_attack.shape) == 1 else _shared_plt_attack
            
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
                if dataset == "ASCAD_desync50" or dataset.startswith("ASCAD"):
                    if leakage == 'ID':
                        key_log_prob[k] += prediction_log[i, AES_Sbox[k ^ int(att_plt[i])]]
                    else:
                        key_log_prob[k] += prediction_log[i, hw[AES_Sbox[k ^ int(att_plt[i])]]]
        
        rank_evol[i] = rk_key(key_log_prob, correct_key)
    
    return rank_evol, key_log_prob

def perform_attacks_ensemble_optimized(nb_traces, ensemble_predictions, plt_attack, correct_key, 
                                     leakage, dataset, nb_attacks=50, shuffle=True, byte=2, 
                                     n_processes=8):
    """
    优化的并行集成攻击函数（使用共享内存）
    """
    
    print(f"🚀 Running {nb_attacks} attacks in parallel using {n_processes} processes (OPTIMIZED)...")
    
    # 将数据转换为适当格式
    ensemble_predictions = np.array(ensemble_predictions, dtype=np.float32)
    plt_attack = np.array(plt_attack, dtype=np.float32)
    
    # 创建共享内存
    shm_predictions = shared_memory.SharedMemory(create=True, size=ensemble_predictions.nbytes)
    shm_plt = shared_memory.SharedMemory(create=True, size=plt_attack.nbytes)
    
    # 复制数据到共享内存
    shared_predictions_array = np.ndarray(ensemble_predictions.shape, dtype=np.float32, buffer=shm_predictions.buf)
    shared_plt_array = np.ndarray(plt_attack.shape, dtype=np.float32, buffer=shm_plt.buf)
    
    shared_predictions_array[:] = ensemble_predictions[:]
    shared_plt_array[:] = plt_attack[:]
    
    # 准备元数据
    metadata = (correct_key, leakage, dataset, byte, nb_traces, shuffle)
    shape_info = (ensemble_predictions.shape, plt_attack.shape)
    
    try:
        # 创建持久进程池
        with mp.Pool(processes=n_processes, 
                    initializer=init_worker,
                    initargs=(shm_predictions.name, shm_plt.name, shape_info, metadata)) as pool:
            
            # 提交所有任务（只传递小的索引，不传递大数组）
            results = pool.map(rank_compute_optimized, range(nb_attacks))
            
            # 收集结果
            all_rk_evol = np.zeros((nb_attacks, nb_traces))
            all_key_log_prob = np.zeros(256)
            
            for i, (rank_evol, key_log_prob) in enumerate(results):
                all_rk_evol[i] = rank_evol
                all_key_log_prob += key_log_prob
    
    finally:
        # 清理共享内存
        shm_predictions.close()
        shm_predictions.unlink()
        shm_plt.close()
        shm_plt.unlink()
    
    # 计算平均值
    avg_rank_evol = np.mean(all_rk_evol, axis=0)
    
    return avg_rank_evol, all_key_log_prob

def perform_attacks_parallel_optimized(nb_traces, predictions, plt_attack, correct_key, 
                                     leakage, dataset, nb_attacks=50, shuffle=True, byte=2, 
                                     n_processes=8):
    """
    优化的并行单模型攻击函数
    """
    # 将单模型转换为集成格式
    ensemble_predictions = [predictions]
    
    return perform_attacks_ensemble_optimized(
        nb_traces, ensemble_predictions, plt_attack, correct_key,
        leakage, dataset, nb_attacks, shuffle, byte, n_processes
    ) 