"""
Enhanced parallel utilities for side-channel attacks
- Fixed indexing issues
- Optimized for high-core CPU utilization
- Improved error handling and logging
"""

import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import random
from functools import partial
import logging
import traceback
from typing import List, Tuple, Optional
from src.utils import AES_Sbox, AES_Sbox_inv, rk_key

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def validate_data_consistency(ensemble_predictions, plt_attack, nb_traces):
    """
    验证数据一致性，防止索引越界
    """
    issues = []
    
    # Check if ensemble_predictions is empty
    if len(ensemble_predictions) == 0:
        issues.append("Empty ensemble_predictions")
        return issues
    
    # Get data sizes
    min_pred_size = min(len(pred) for pred in ensemble_predictions)
    max_pred_size = max(len(pred) for pred in ensemble_predictions)
    plt_size = len(plt_attack)
    
    logger.info(f"Data validation:")
    logger.info(f"  Prediction sizes: {min_pred_size} - {max_pred_size}")
    logger.info(f"  Plaintext size: {plt_size}")
    logger.info(f"  Requested traces: {nb_traces}")
    
    # Check for size mismatches
    if min_pred_size != max_pred_size:
        issues.append(f"Inconsistent prediction sizes: {min_pred_size} - {max_pred_size}")
    
    # Check if we have enough data
    available_traces = min(min_pred_size, plt_size)
    if nb_traces > available_traces:
        issues.append(f"Requested {nb_traces} traces but only {available_traces} available")
    
    return issues

def safe_random_indices(data_size: int, nb_traces: int, attack_idx: int, model_idx: int) -> np.ndarray:
    """
    安全地生成随机索引，防止越界
    """
    # Ensure we don't request more traces than available
    actual_traces = min(nb_traces, data_size)
    
    # Set reproducible random seed
    np.random.seed(attack_idx * 1000 + model_idx)
    
    # Generate indices
    if actual_traces == data_size:
        return np.arange(data_size)
    else:
        return np.random.choice(data_size, size=actual_traces, replace=False)

def rank_compute_single_attack_enhanced(args):
    """
    Enhanced single attack computation with robust error handling
    """
    try:
        (ensemble_predictions, plt_attack, correct_key, leakage, dataset, 
         byte, nb_traces, attack_idx, shuffle) = args
        
        hw = [bin(x).count("1") for x in range(256)]
        total_num_model = len(ensemble_predictions)
        
        # Check for empty ensemble
        if total_num_model == 0:
            logger.warning(f"Attack {attack_idx}: Empty ensemble, returning worst case")
            return np.full(nb_traces, 255.0), np.zeros(256)
        
        # Validate data sizes and get safe trace count
        min_pred_size = min(len(pred) for pred in ensemble_predictions)
        plt_size = len(plt_attack)
        available_traces = min(min_pred_size, plt_size)
        
        if nb_traces > available_traces:
            logger.warning(f"Attack {attack_idx}: Requested {nb_traces} traces, "
                         f"but only {available_traces} available. Using {available_traces}.")
            actual_traces = available_traces
        else:
            actual_traces = nb_traces
        
        # Prepare data safely
        ensemble_att_pred = []
        ensemble_att_plt = []
        
        for model_idx in range(total_num_model):
            pred_size = len(ensemble_predictions[model_idx])
            
            if shuffle:
                # Generate safe random indices
                indices = safe_random_indices(
                    min(pred_size, plt_size), actual_traces, attack_idx, model_idx
                )
                
                # Safely extract data
                att_pred = ensemble_predictions[model_idx][indices]
                if len(plt_attack.shape) == 1:
                    att_plt = plt_attack[indices]
                else:
                    att_plt = plt_attack[indices] if len(indices) <= len(plt_attack) else plt_attack[:len(indices)]
            else:
                # Sequential selection
                att_pred = ensemble_predictions[model_idx][:actual_traces]
                if len(plt_attack.shape) == 1:
                    att_plt = plt_attack[:actual_traces]
                else:
                    att_plt = plt_attack[:actual_traces]
            
            ensemble_att_pred.append(att_pred)
            ensemble_att_plt.append(att_plt)
        
        # Compute rank evolution
        key_log_prob = np.zeros(256)
        (nb_traces_actual, nb_hyp) = ensemble_att_pred[0].shape
        rank_evol = np.full(nb_traces_actual, 255.0)
        
        for i in range(nb_traces_actual):
            for model_idx in range(total_num_model):
                prediction = ensemble_att_pred[model_idx]
                prediction_log = np.log(prediction + 1e-40)
                
                att_plt = ensemble_att_plt[model_idx]
                
                for k in range(256):
                    try:
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
                                plt_val = int(att_plt[i]) if hasattr(att_plt[i], '__int__') else int(att_plt[i, 0])
                                key_log_prob[k] += prediction_log[i, AES_Sbox[k ^ plt_val]]
                            else:
                                plt_val = int(att_plt[i]) if hasattr(att_plt[i], '__int__') else int(att_plt[i, 0])
                                key_log_prob[k] += prediction_log[i, hw[AES_Sbox[k ^ plt_val]]]
                    except (IndexError, TypeError) as e:
                        logger.error(f"Attack {attack_idx}, model {model_idx}, trace {i}, key {k}: {str(e)}")
                        continue
            
            rank_evol[i] = rk_key(key_log_prob, correct_key)
        
        return rank_evol, key_log_prob
        
    except Exception as e:
        logger.error(f"Attack {attack_idx} failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return worst case scenario
        return np.full(nb_traces, 255.0), np.zeros(256)

def perform_attacks_ensemble_parallel_enhanced(nb_traces, ensemble_predictions, plt_attack, 
                                             correct_key, leakage, dataset, nb_attacks=1, 
                                             shuffle=True, byte=2, n_processes=None, 
                                             max_workers_per_core=2):
    """
    Enhanced parallel ensemble attack with optimized CPU utilization
    
    Args:
        max_workers_per_core: Multiplier for CPU cores to create more workers
    """
    
    # Validate inputs first
    validation_issues = validate_data_consistency(ensemble_predictions, plt_attack, nb_traces)
    if validation_issues:
        logger.error("Data validation failed:")
        for issue in validation_issues:
            logger.error(f"  - {issue}")
        # Return worst case
        return np.full(nb_traces, 255.0), np.zeros(256)
    
    # Optimize process count for high-core systems
    cpu_count = mp.cpu_count()
    if n_processes is None:
        # For 128-core systems, use more aggressive parallelization
        n_processes = min(nb_attacks, cpu_count * max_workers_per_core)
        # But cap it at reasonable limit to avoid overhead
        n_processes = min(n_processes, 64)
    
    logger.info(f"🚀 Enhanced parallel attack:")
    logger.info(f"  CPU cores: {cpu_count}")
    logger.info(f"  Process count: {n_processes}")
    logger.info(f"  Attacks: {nb_attacks}")
    logger.info(f"  Traces per attack: {nb_traces}")
    logger.info(f"  Models in ensemble: {len(ensemble_predictions)}")
    
    # Prepare arguments
    args_list = []
    for attack_idx in range(nb_attacks):
        args = (ensemble_predictions, plt_attack, correct_key, leakage, dataset,
                byte, nb_traces, attack_idx, shuffle)
        args_list.append(args)
    
    # Execute attacks
    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    all_key_log_prob = np.zeros(256)
    
    if n_processes == 1:
        # Single process mode for debugging
        logger.info("Running in single process mode...")
        for i, args in enumerate(tqdm(args_list, desc="Processing attacks")):
            rank_evol, key_log_prob = rank_compute_single_attack_enhanced(args)
            all_rk_evol[i] = rank_evol
            all_key_log_prob += key_log_prob
    else:
        # Multi-process execution with optimized settings
        logger.info(f"Running in parallel mode with {n_processes} processes...")
        
        try:
            # Use spawn method for better reliability on high-core systems
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_processes) as pool:
                # Submit all tasks
                results = []
                for args in args_list:
                    result = pool.apply_async(rank_compute_single_attack_enhanced, (args,))
                    results.append(result)
                
                # Collect results with progress bar
                for i, result in enumerate(tqdm(results, desc="Collecting results")):
                    try:
                        rank_evol, key_log_prob = result.get(timeout=300)  # 5 min timeout
                        all_rk_evol[i] = rank_evol
                        all_key_log_prob += key_log_prob
                    except mp.TimeoutError:
                        logger.error(f"Attack {i} timed out")
                        all_rk_evol[i] = np.full(nb_traces, 255.0)
                    except Exception as e:
                        logger.error(f"Attack {i} failed: {str(e)}")
                        all_rk_evol[i] = np.full(nb_traces, 255.0)
                        
        except Exception as e:
            logger.error(f"Parallel execution failed: {str(e)}")
            logger.info("Falling back to single process mode...")
            # Fallback to single process
            for i, args in enumerate(tqdm(args_list, desc="Processing attacks (fallback)")):
                rank_evol, key_log_prob = rank_compute_single_attack_enhanced(args)
                all_rk_evol[i] = rank_evol
                all_key_log_prob += key_log_prob
    
    # Calculate average
    avg_rank_evol = np.mean(all_rk_evol, axis=0)
    
    logger.info(f"✅ Parallel attack completed:")
    logger.info(f"  Final average GE: {avg_rank_evol[-1]:.3f}")
    logger.info(f"  Successful attacks: {np.sum(all_rk_evol[:, -1] < 255)}/{nb_attacks}")
    
    return avg_rank_evol, all_key_log_prob

def perform_attacks_single_model_parallel_enhanced(nb_traces, predictions, plt_attack, 
                                                 correct_key, leakage, dataset, 
                                                 nb_attacks=1, shuffle=True, byte=2, 
                                                 n_processes=None):
    """
    Enhanced parallel single model attack
    """
    # Convert to ensemble format
    ensemble_predictions = [predictions]
    
    return perform_attacks_ensemble_parallel_enhanced(
        nb_traces, ensemble_predictions, plt_attack, correct_key,
        leakage, dataset, nb_attacks, shuffle, byte, n_processes
    )

def adaptive_process_count(nb_attacks: int, cpu_count: int, ensemble_size: int) -> int:
    """
    Adaptively determine optimal process count based on workload
    """
    # Base calculation
    base_processes = min(nb_attacks, cpu_count)
    
    # Adjust based on ensemble size (larger ensembles need more memory per process)
    if ensemble_size > 50:
        memory_factor = 0.8
    elif ensemble_size > 20:
        memory_factor = 1.0
    else:
        memory_factor = 1.5
    
    # For high-core systems (>64 cores), be more aggressive
    if cpu_count >= 64:
        core_factor = 1.5
    elif cpu_count >= 32:
        core_factor = 1.2
    else:
        core_factor = 1.0
    
    optimal_processes = int(base_processes * memory_factor * core_factor)
    
    # Cap at reasonable limits
    optimal_processes = min(optimal_processes, cpu_count * 2, 96)
    optimal_processes = max(optimal_processes, 1)
    
    logger.info(f"Adaptive process count: {optimal_processes} "
               f"(CPU: {cpu_count}, Attacks: {nb_attacks}, Ensemble: {ensemble_size})")
    
    return optimal_processes

def benchmark_parallel_performance(ensemble_predictions, plt_attack, correct_key, 
                                 leakage, dataset, byte=2, nb_traces=1000):
    """
    Benchmark different parallel configurations to find optimal settings
    """
    cpu_count = mp.cpu_count()
    test_processes = [1, cpu_count//4, cpu_count//2, cpu_count, cpu_count*2]
    test_processes = [p for p in test_processes if p >= 1]
    
    logger.info(f"🔬 Benchmarking parallel performance on {cpu_count}-core system...")
    
    benchmark_results = {}
    
    for n_proc in test_processes:
        logger.info(f"Testing with {n_proc} processes...")
        
        import time
        start_time = time.time()
        
        try:
            avg_rank_evol, _ = perform_attacks_ensemble_parallel_enhanced(
                nb_traces=nb_traces,
                ensemble_predictions=ensemble_predictions[:5],  # Use subset for benchmark
                plt_attack=plt_attack,
                correct_key=correct_key,
                leakage=leakage,
                dataset=dataset,
                nb_attacks=10,  # Small number for benchmark
                shuffle=True,
                byte=byte,
                n_processes=n_proc
            )
            
            elapsed_time = time.time() - start_time
            final_ge = avg_rank_evol[-1]
            
            benchmark_results[n_proc] = {
                'time': elapsed_time,
                'final_ge': final_ge,
                'speedup': benchmark_results[1]['time'] / elapsed_time if 1 in benchmark_results else 1.0
            }
            
            logger.info(f"  {n_proc} processes: {elapsed_time:.2f}s, GE: {final_ge:.3f}")
            
        except Exception as e:
            logger.error(f"  {n_proc} processes failed: {str(e)}")
            benchmark_results[n_proc] = {'time': float('inf'), 'final_ge': 255.0, 'speedup': 0.0}
    
    # Find optimal configuration
    valid_results = {k: v for k, v in benchmark_results.items() if v['time'] < float('inf')}
    if valid_results:
        optimal_processes = min(valid_results.keys(), key=lambda k: valid_results[k]['time'])
        logger.info(f"🏆 Optimal configuration: {optimal_processes} processes")
        logger.info(f"   Speedup: {valid_results[optimal_processes]['speedup']:.2f}x")
    else:
        optimal_processes = 1
        logger.warning("All parallel configurations failed, using single process")
    
    return optimal_processes, benchmark_results 