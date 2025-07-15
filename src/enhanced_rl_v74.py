import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import logging
import time
import os
from src.enhanced_utils_parallel import perform_attacks_ensemble_parallel_enhanced, adaptive_process_count
from src.utils import NTGE_fn

# 导入Q表可视化器
try:
    from src.q_table_visualizer import QTableVisualizer
except ImportError:
    # 如果导入失败，创建一个空的类避免错误
    class QTableVisualizer:
        def __init__(self, *args, **kwargs):
            pass
        def record_q_values(self, *args, **kwargs):
            pass
        def generate_comprehensive_report(self):
            pass

# Enhanced Experience Replay for Global Strategy Learning
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class PrioritizedReplayMemory:
    """优先经验回放，提高学习效率"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        # Importance-sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, torch.FloatTensor(weights)
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.memory)

class GlobalStrategyDQN(nn.Module):
    """
    全局策略感知的DQN网络
    - 直接输出每个模型的选择概率/价值
    - 具备全局组合感知能力
    """
    def __init__(self, total_models, hidden_dim=512):
        super(GlobalStrategyDQN, self).__init__()
        self.total_models = total_models
        
        # Multi-head attention for global awareness
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1
        )
        
        # Feature extraction layers
        self.input_projection = nn.Linear(total_models, hidden_dim)
        self.feature_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        # Global context processing
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Output layers for each model
        self.model_values = nn.Linear(hidden_dim + hidden_dim // 4, total_models)
        self.combination_value = nn.Linear(hidden_dim + hidden_dim // 4, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        # Input projection
        x = F.relu(self.input_projection(state))
        x = self.dropout(x)
        
        # Feature extraction with residual connections
        for layer in self.feature_layers:
            residual = x
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        # Self-attention for global awareness
        if len(x.shape) == 2:
            x_unsqueezed = x.unsqueeze(0)  # Add sequence dimension
        else:
            x_unsqueezed = x
            
        attn_output, _ = self.self_attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = attn_output.squeeze(0) if len(x.shape) == 2 else attn_output
        
        # Global context
        if len(x.shape) == 2:
            # x is [batch_size, hidden_dim]
            global_features = self.global_context(x.mean(dim=0, keepdim=True))  # [1, hidden_dim//4]
            global_features = global_features.expand(x.shape[0], -1)            # [batch_size, hidden_dim//4]
        else:
            # x is [hidden_dim]
            global_features = self.global_context(x)                             # [hidden_dim//4]
            global_features = global_features.unsqueeze(0)                       # [1, hidden_dim//4]
        
        # Combine local and global features
        combined_features = torch.cat([x, global_features], dim=-1)
        
        # Output model values and combination value
        model_values = self.model_values(combined_features)
        combination_value = self.combination_value(combined_features)
        
        return model_values, combination_value

class EnhancedModelSelector:
    """
    增强的模型选择器：
    1. 全局策略感知
    2. 二进制向量表示
    3. 完整的可解释性分析
    4. 高效的并行处理
    """
    
    def __init__(self, device, total_models=100, max_models=15):
        self.device = device
        self.total_models = total_models
        self.max_models = max_models
        
        # Enhanced DQN parameters
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.01
        self.EPS_DECAY = 2000
        self.TAU = 0.001
        self.LR = 5e-4
        
        # Initialize networks
        self.policy_net = GlobalStrategyDQN(total_models).to(device)
        self.target_net = GlobalStrategyDQN(total_models).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Enhanced optimizer with scheduler
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.LR, 
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500, gamma=0.95
        )
        
        # Prioritized experience replay
        self.memory = PrioritizedReplayMemory(20000)
        self.steps_done = 0
        
        # Training metrics for analysis
        self.training_metrics = {
            'rewards': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': [],
            'model_selection_frequency': np.zeros(total_models),
            'episode_lengths': [],
            'final_ge_values': []
        }
        
        # Initialize Q-table visualizer
        self.q_visualizer = None
        self.enable_q_visualization = True
        
    def get_binary_state(self, selected_models: List[int]) -> torch.Tensor:
        """
        获取二进制状态表示
        Args:
            selected_models: 已选择的模型索引列表
        Returns:
            二进制状态向量 (1表示选择，0表示未选择)
        """
        state = torch.zeros(self.total_models, device=self.device)
        if selected_models:
            state[selected_models] = 1.0
        return state
        
    def select_action_epsilon_greedy(self, state: torch.Tensor, step: int, 
                                   available_models: List[int], 
                                   selected_models: List[int],
                                   is_training: bool = True) -> int:
        """
        使用ε-贪婪策略选择下一个要添加的模型
        """
        # Calculate epsilon
        if is_training:
            eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                  math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            self.training_metrics['epsilon_values'].append(eps)
        else:
            eps = 0.0
            
        # Get model values from network
        with torch.no_grad():
            model_values, combination_value = self.policy_net(state.unsqueeze(0))
            model_values = model_values.squeeze()
            
            # Record Q-values for analysis
            if is_training:
                self.training_metrics['q_values'].append(model_values.cpu().numpy())
        
        # Epsilon-greedy selection among available models
        if random.random() < eps and is_training:
            # Exploration: random selection from available models
            action = random.choice(available_models)
        else:
            # Exploitation: select model with highest value among available
            available_values = model_values[available_models]
            best_idx = available_values.argmax().item()
            action = available_models[best_idx]
        
        # Record Q-values for visualization if enabled
        if self.enable_q_visualization and self.q_visualizer is not None and is_training:
            # 这里先记录一个占位奖励，后续会更新
            self.q_visualizer.record_q_values(
                episode=step // 50,  # 粗略估计episode
                state=state,
                q_values=model_values,
                selected_action=action,
                reward=0.0  # 占位值，后续更新
            )
            
        return action
        
    def calculate_reward(self, prev_ge: float, new_ge: float, 
                        prev_models: List[int], new_models: List[int],
                        episode_step: int, max_steps: int = 50,
                        prev_ntge: float = None, new_ntge: float = None) -> float:
        """
        计算奖励函数（以NTGE为核心决策指标）
        
        Args:
            prev_ge: 前一个状态的GE值
            new_ge: 新状态的GE值
            prev_models: 前一个状态的模型列表
            new_models: 新状态的模型列表
            episode_step: 当前步数
            max_steps: 最大步数
            prev_ntge: 前一个状态的NTGE值
            new_ntge: 新状态的NTGE值
        """
        # 核心目标：最小化NTGE（达到GE=0的最少轨迹数）
        if new_ntge is not None and new_ntge < float('inf'):
            # 成功达到GE=0，NTGE越小奖励越高
            success_reward = 1000.0  # 🔧 基础成功奖励（降低避免过度优化）
            
            # NTGE效率奖励：更精细的分级，突出最优模型
            if new_ntge <= 1500:
                ntge_efficiency = 800.0  # 极优秀
            elif new_ntge <= 1700:
                ntge_efficiency = 600.0  # 很优秀  
            elif new_ntge <= 1900:
                ntge_efficiency = 400.0  # 优秀
            elif new_ntge <= 2000:
                ntge_efficiency = 200.0  # 良好
            elif new_ntge <= 5000:
                ntge_efficiency = 100.0  # 一般
            else:
                ntge_efficiency = 50.0   # 较差
                
            # NTGE改进奖励（只有在有比较对象时才计算）
            if prev_ntge is not None and prev_ntge != float('inf'):
                ntge_improvement = max(0, (prev_ntge - new_ntge) * 0.1)
            else:
                ntge_improvement = 0
            
            # 模型效率奖励：使用更少的模型获得更高奖励
            efficiency_bonus = max(0, (self.max_models - len(new_models)) * 50.0)
            
            # 速度奖励：更快找到解决方案获得更高奖励
            speed_bonus = max(0, (max_steps - episode_step) * 10.0)
            
            total_reward = success_reward + ntge_efficiency + ntge_improvement + efficiency_bonus + speed_bonus
            return total_reward
        
        # 如果NTGE为无穷大（无法达到GE=0），基于GE改进计算奖励
        elif new_ntge == float('inf'):
            # 🔧 无法达到GE=0时，基于GE给予合理奖励
            
            # GE改进奖励（主要指标）
            ge_improvement = prev_ge - new_ge
            ge_reward = ge_improvement * 10.0  # 降低权重避免过度优化
            
            # 基于GE绝对值的奖励（避免所有奖励都是负数）
            if new_ge < 1.0:
                base_reward = 200.0  # 很好
            elif new_ge < 5.0:
                base_reward = 100.0  # 良好
            elif new_ge < 10.0:
                base_reward = 50.0   # 一般
            elif new_ge < 50.0:
                base_reward = 10.0   # 较差
            else:
                base_reward = -10.0  # 很差
            
            # 模型效率奖励：使用更少的模型获得更高奖励
            efficiency_bonus = max(0, (self.max_models - len(new_models)) * 5.0)
            
            # 惩罚：没有改善
            if new_ge >= prev_ge:
                stagnation_penalty = -20.0
            else:
                stagnation_penalty = 0
            
            # 综合奖励
            reward = base_reward + ge_reward + efficiency_bonus + stagnation_penalty
            
        else:
            # 如果没有NTGE信息，使用传统的GE-based奖励
            ge_improvement = prev_ge - new_ge
            
            # 接近目标的奖励
            if new_ge < 10:
                proximity_reward = (10 - new_ge) * 10.0
            elif new_ge < 50:
                proximity_reward = (50 - new_ge) * 1.0
            else:
                proximity_reward = 0
            
            # 惩罚
            model_penalty = len(new_models) * 2.0
            stagnation_penalty = -10.0 if new_ge >= prev_ge else 0
            
            reward = ge_improvement * 20.0 + proximity_reward - model_penalty + stagnation_penalty
        
        return reward

    def evaluate_combination_quality(self, ge: float, ntge: float, model_count: int) -> float:
        """
        评估模型组合质量（以NTGE为核心指标）
        
        Args:
            ge: 猜测熵值
            ntge: 达到GE=0所需轨迹数
            model_count: 使用的模型数量
            
        Returns:
            综合质量评分（越高越好）
        """
        # NTGE分数：NTGE越小越好（核心指标）
        if ntge != float('inf'):
            ntge_score = max(0, 100 - ntge * 0.02)  # NTGE=0得100分，NTGE=5000得0分
        else:
            # 如果NTGE为无穷大，基于GE计算分数
            ntge_score = max(0, 50 - ge * 5)  # GE=0得50分，GE=10得0分
        
        # 模型效率分数：使用更少模型更好
        efficiency_score = max(0, 30 - model_count * 2)  # 0个模型得30分，15个模型得0分
        
        # 综合评分（NTGE权重最高）
        total_score = ntge_score * 0.8 + efficiency_score * 0.2
        
        return total_score

    def should_update_best_result(self, current_best_ge: float, current_best_ntge: float,
                                new_ge: float, new_ntge: float, 
                                current_best_score: float = None) -> bool:
        """
        判断是否应该更新最佳结果（以NTGE为核心）
        
        Args:
            current_best_ge: 当前最佳GE
            current_best_ntge: 当前最佳NTGE
            new_ge: 新的GE
            new_ntge: 新的NTGE
            current_best_score: 当前最佳综合评分
            
        Returns:
            是否应该更新
        """
        # 计算新的综合评分
        new_score = self.evaluate_combination_quality(new_ge, new_ntge, 0)  # model_count暂时设为0
        
        # 如果当前最佳评分未提供，使用简单的比较逻辑
        if current_best_score is None:
            # 优先考虑NTGE
            if new_ntge < float('inf') and current_best_ntge == float('inf'):
                return True  # 从无法达到GE=0到可以达到
            elif new_ntge < float('inf') and current_best_ntge < float('inf'):
                return new_ntge < current_best_ntge  # 都达到GE=0，选择NTGE更小的
            elif new_ntge == float('inf') and current_best_ntge == float('inf'):
                # 都无法达到GE=0，比较GE
                return new_ge < current_best_ge
            else:
                return False  # 当前可以达到GE=0，新的无法达到
        else:
            # 使用综合评分比较
            return new_score > current_best_score
        
    def optimize_model(self):
        """优化模型（使用优先经验回放）"""
        if len(self.memory) < self.BATCH_SIZE:
            return
            
        # Sample from prioritized replay
        transitions, indices, weights = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Prepare batch data
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Handle next states (some might be None for terminal states)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Current Q-values
        model_values, combination_values = self.policy_net(state_batch)
        current_q_values = model_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
        # Next Q-values
        next_q_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            if non_final_mask.any():
                next_model_values, _ = self.target_net(non_final_next_states)
                next_q_values[non_final_mask] = next_model_values.max(1)[0]
        
        # Expected Q-values
        expected_q_values = reward_batch + (self.GAMMA * next_q_values * ~done_batch)
        
        # Compute loss with importance sampling weights
        td_errors = current_q_values - expected_q_values.detach()
        loss = (weights.to(self.device) * (td_errors ** 2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        priorities = (abs(td_errors) + 1e-6).detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        # Record training metrics
        self.training_metrics['losses'].append(loss.item())
        
    def update_target_network(self):
        """软更新目标网络"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                policy_net_state_dict[key] * self.TAU + \
                target_net_state_dict[key] * (1 - self.TAU)
                
        self.target_net.load_state_dict(target_net_state_dict)
        
    def save_analysis_data(self, save_path: str, episode: int):
        """保存完整的分析数据"""
        analysis_data = {
            'episode': episode,
            'training_metrics': self.training_metrics,
            'network_state': {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'hyperparameters': {
                'total_models': self.total_models,
                'max_models': self.max_models,
                'batch_size': self.BATCH_SIZE,
                'gamma': self.GAMMA,
                'learning_rate': self.LR,
                'tau': self.TAU
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as pickle for complete data
        with open(f"{save_path}/analysis_data_episode_{episode}.pkl", 'wb') as f:
            pickle.dump(analysis_data, f)
            
        # Save training metrics as JSON for easy access
        metrics_json = {
            'rewards': self.training_metrics['rewards'],
            'losses': self.training_metrics['losses'],
            'epsilon_values': self.training_metrics['epsilon_values'],
            'episode_lengths': self.training_metrics['episode_lengths'],
            'final_ge_values': self.training_metrics['final_ge_values'],
            'model_selection_frequency': self.training_metrics['model_selection_frequency'].tolist()
        }
        
        with open(f"{save_path}/training_metrics_episode_{episode}.json", 'w') as f:
            json.dump(metrics_json, f, indent=2)
            
    def generate_visualizations(self, save_path: str, episode: int):
        """生成完整的可视化分析"""
        
        # 1. Training Progress
        self._plot_training_progress(save_path, episode)
        
        # 2. Q-value Analysis
        self._plot_q_value_analysis(save_path, episode)
        
        # 3. Model Selection Frequency
        self._plot_model_selection_frequency(save_path, episode)
        
        # 4. Strategy Evolution
        self._plot_strategy_evolution(save_path, episode)
        
    def _plot_training_progress(self, save_path: str, episode: int):
        """绘制训练进度"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        if self.training_metrics['rewards']:
            ax1.plot(self.training_metrics['rewards'], alpha=0.7)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.grid(True, alpha=0.3)
        
        # Losses
        if self.training_metrics['losses']:
            ax2.plot(self.training_metrics['losses'], alpha=0.7, color='red')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        
        # Epsilon values
        if self.training_metrics['epsilon_values']:
            ax3.plot(self.training_metrics['epsilon_values'], alpha=0.7, color='green')
            ax3.set_title('Exploration Rate (Epsilon)')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Epsilon')
            ax3.grid(True, alpha=0.3)
            
        # Episode lengths
        if self.training_metrics['episode_lengths']:
            ax4.plot(self.training_metrics['episode_lengths'], alpha=0.7, color='purple')
            ax4.set_title('Episode Lengths')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Steps')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_progress_episode_{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_q_value_analysis(self, save_path: str, episode: int):
        """绘制Q值分析"""
        if not self.training_metrics['q_values']:
            return
            
        q_values_array = np.array(self.training_metrics['q_values'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Q-value distribution
        ax1.hist(q_values_array.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Q-value Distribution')
        ax1.set_xlabel('Q-value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Q-value evolution over time
        if len(q_values_array) > 10:
            mean_q_values = np.mean(q_values_array, axis=1)
            ax2.plot(mean_q_values, alpha=0.8)
            ax2.set_title('Average Q-value Evolution')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Average Q-value')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/q_value_analysis_episode_{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_model_selection_frequency(self, save_path: str, episode: int):
        """绘制模型选择频率"""
        plt.figure(figsize=(15, 6))
        
        model_indices = range(self.total_models)
        frequencies = self.training_metrics['model_selection_frequency']
        
        plt.bar(model_indices, frequencies, alpha=0.7)
        plt.title('Model Selection Frequency During Training')
        plt.xlabel('Model Index')
        plt.ylabel('Selection Count')
        plt.grid(True, alpha=0.3)
        
        # Highlight top models
        top_k = 10
        top_indices = np.argsort(frequencies)[-top_k:]
        for idx in top_indices:
            plt.bar(idx, frequencies[idx], color='red', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/model_selection_frequency_episode_{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_strategy_evolution(self, save_path: str, episode: int):
        """绘制策略演化"""
        if len(self.training_metrics['final_ge_values']) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final GE values over episodes
        ax1.plot(self.training_metrics['final_ge_values'], marker='o', alpha=0.7)
        ax1.set_title('Final GE Values Over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Final GE')
        ax1.grid(True, alpha=0.3)
        
        # Running average
        if len(self.training_metrics['final_ge_values']) > 10:
            window = min(10, len(self.training_metrics['final_ge_values']) // 4)
            running_avg = np.convolve(self.training_metrics['final_ge_values'], 
                                    np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.training_metrics['final_ge_values'])), 
                    running_avg, color='red', linewidth=2, label=f'Running Avg ({window})')
            ax1.legend()
        
        # Strategy convergence
        if self.training_metrics['rewards']:
            ax2.plot(np.cumsum(self.training_metrics['rewards']), alpha=0.7, color='green')
            ax2.set_title('Cumulative Rewards')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Cumulative Reward')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/strategy_evolution_episode_{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def global_binary_strategy_selection(self, ensemble_predictions, all_ind_GE, 
                                       all_ind_NTGE=None, target_models=15, 
                                       perform_attacks_ensemble=None, 
                                       nb_traces_attacks=1000, plt_val=None, 
                                       correct_key=None, dataset="ASCAD", 
                                       nb_attacks=50, leakage="HW", byte=2,
                                       num_episodes=100, max_episode_length=None):
        """
        全局二进制策略模型选择
        - 直接学习最优的二进制向量表示
        - 具备全局选择意识
        - 完整的可解释性分析
        
        Args:
            target_models: 目标模型数量
            num_episodes: 训练轮数
            max_episode_length: 最大单轮长度（None为自动设置）
        """
        
        logger.info(f"🎯 开始全局二进制策略DQN训练")
        logger.info(f"  总模型数: {self.total_models}")
        logger.info(f"  目标选择: {target_models}")
        logger.info(f"  训练轮数: {num_episodes}")
        
        if max_episode_length is None:
            max_episode_length = min(target_models * 3, 50)
        
        # Initialize performance tracking
        # 🔧 修复: 确保至少有一个最佳单模型作为基线
        if all_ind_NTGE is not None:
            valid_ntge_indices = np.where(all_ind_NTGE != float('inf'))[0]
            if len(valid_ntge_indices) > 0:
                best_single_model = valid_ntge_indices[np.argmin(all_ind_NTGE[valid_ntge_indices])]
                best_ge = all_ind_GE[best_single_model]
                best_ntge = all_ind_NTGE[best_single_model]
            else:
                best_single_model = np.argmin(all_ind_GE)
                best_ge = all_ind_GE[best_single_model]
                best_ntge = float('inf')
        else:
            best_single_model = np.argmin(all_ind_GE)
            best_ge = all_ind_GE[best_single_model]
            best_ntge = float('inf')
        
        # 确保至少有一个模型作为基线结果
        best_combination = [best_single_model]
        best_ge_evolution = [best_ge]
        
        print(f"🎯 初始化基线模型:")
        print(f"   模型索引: {best_single_model}")
        print(f"   基线GE: {best_ge:.4f}")
        print(f"   基线NTGE: {best_ntge if best_ntge != float('inf') else '∞'}")
        
        # Episode loop
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            # Run one training episode
            episode_results = self._run_global_strategy_episode(
                ensemble_predictions, all_ind_GE, target_models,
                perform_attacks_ensemble, nb_traces_attacks, plt_val,
                correct_key, dataset, nb_attacks, leakage, byte,
                episode, max_episode_length, all_ind_NTGE, is_training=True
            )
            
            # Update best results using multi-objective evaluation
            current_best_score = self.evaluate_combination_quality(best_ge, best_ntge, len(best_combination))
            new_score = self.evaluate_combination_quality(episode_results['final_ge'], 
                                                        episode_results['final_ntge'], 
                                                        len(episode_results['selected_models']))
            
            if self.should_update_best_result(best_ge, best_ntge, 
                                              episode_results['final_ge'], 
                                              episode_results['final_ntge'], 
                                              current_best_score):
                best_ge = episode_results['final_ge']
                best_ntge = episode_results['final_ntge']
                best_combination = episode_results['selected_models'].copy()
                best_ge_evolution = episode_results['ge_evolution'].copy()
                best_ntge_evolution = episode_results['ntge_evolution'].copy()
                print(f"🎉 New best combination found!")
                print(f"   Models: {best_combination}")
                print(f"   Final GE: {best_ge:.4f}")
                print(f"   Final NTGE: {best_ntge if best_ntge != float('inf') else '∞'}")
                print(f"   Model count: {len(best_combination)}")
                print(f"   Quality Score: {new_score:.2f} (prev: {current_best_score:.2f})")
            
            # Record episode metrics
            self.training_metrics['rewards'].append(episode_results['total_reward'])
            self.training_metrics['episode_lengths'].append(episode_results['episode_length'])
            self.training_metrics['final_ge_values'].append(episode_results['final_ge'])
            
            # Update model selection frequency
            for model_idx in episode_results['selected_models']:
                self.training_metrics['model_selection_frequency'][model_idx] += 1
            
            # Optimize model
            if len(self.memory) >= self.BATCH_SIZE:
                for _ in range(5):  # Multiple updates per episode
                    self.optimize_model()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.update_target_network()
                
            episode_time = time.time() - episode_start_time
            
            print(f"Episode {episode + 1} completed in {episode_time:.2f}s")
            print(f"  Final GE: {episode_results['final_ge']:.4f}")
            print(f"  Final NTGE: {episode_results['final_ntge'] if episode_results['final_ntge'] != float('inf') else '∞'}")
            print(f"  Total reward: {episode_results['total_reward']:.3f}")
            print(f"  Models selected: {len(episode_results['selected_models'])}")
            print(f"  Current best GE: {best_ge:.4f}")
            
            # Periodic analysis and visualization
            if (episode + 1) % 20 == 0:
                print(f"📊 Generating intermediate analysis...")
                try:
                    analysis_dir = f"analysis_episode_{episode + 1}"
                    os.makedirs(analysis_dir, exist_ok=True)
                    self.save_analysis_data(analysis_dir, episode + 1)
                    self.generate_visualizations(analysis_dir, episode + 1)
                except Exception as e:
                    print(f"⚠️ Analysis generation failed: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"🏁 训练完成")
        print(f"{'='*60}")
        
        # Final evaluation with greedy policy
        print(f"📊 最终贪婪策略评估...")
        final_results = self._run_global_strategy_episode(
            ensemble_predictions, all_ind_GE, target_models,
            perform_attacks_ensemble, nb_traces_attacks, plt_val,
            correct_key, dataset, nb_attacks, leakage, byte,
            num_episodes, max_episode_length, all_ind_NTGE, is_training=False
        )
        
        # 更新最佳结果（使用多目标评估）
        current_best_score = self.evaluate_combination_quality(best_ge, best_ntge, len(best_combination))
        final_score = self.evaluate_combination_quality(final_results['final_ge'], 
                                                      final_results['final_ntge'], 
                                                      len(final_results['selected_models']))
        
        if self.should_update_best_result(best_ge, best_ntge, 
                                          final_results['final_ge'], 
                                          final_results['final_ntge'], 
                                          current_best_score):
            best_ge = final_results['final_ge']
            best_ntge = final_results['final_ntge']
            best_combination = final_results['selected_models'].copy()
            best_ge_evolution = final_results['ge_evolution'].copy()
            best_ntge_evolution = final_results['ntge_evolution'].copy()
            print(f"🎉 Final greedy policy achieved new best!")
            print(f"   Quality Score: {final_score:.2f} (prev: {current_best_score:.2f})")
        
        # 🔧 确保返回有效结果：如果best_combination为空，返回基线单模型
        if not best_combination:
            print(f"⚠️ 训练未找到更好组合，返回基线单模型")
            best_combination = [best_single_model]
            best_ge = all_ind_GE[best_single_model] 
            best_ntge = all_ind_NTGE[best_single_model] if all_ind_NTGE is not None else float('inf')
            best_ge_evolution = [best_ge]
        
        print(f"\n🏆 最佳结果:")
        print(f"   选择模型: {best_combination}")
        print(f"   模型数量: {len(best_combination)}")
        print(f"   最终GE: {best_ge:.4f}")
        print(f"   最终NTGE: {best_ntge if best_ntge != float('inf') else '∞'}")
        
        if best_ge == 0:
            print(f"   🎯 成功达到 GE=0!")
            if best_ntge != float('inf'):
                print(f"   最少轨迹数: {best_ntge:.0f}")
            print(f"   收敛性: 完美")
        elif best_ge < 10:
            print(f"   收敛性: 良好")
        else:
            print(f"   收敛性: 需改进")
        
        # 生成Q表可视化报告
        if self.enable_q_visualization and self.q_visualizer is not None:
            try:
                print(f"\n📊 生成Q表可视化报告...")
                self.q_visualizer.generate_comprehensive_report()
                print(f"✅ Q表可视化报告生成完成")
            except Exception as e:
                print(f"⚠️ Q表可视化报告生成失败: {str(e)}")
        
        return best_combination, best_ge_evolution
        
    def _run_global_strategy_episode(self, ensemble_predictions, all_ind_GE, target_models,
                                   perform_attacks_ensemble, nb_traces_attacks, plt_val,
                                   correct_key, dataset, nb_attacks, leakage, byte,
                                   episode, max_steps, all_ind_NTGE=None, is_training=True):
        """
        运行单个全局策略回合（优化版本，包含NTGE计算）
        """
        
        selected_models = []
        available_models = list(range(self.total_models))
        ge_evolution = []
        ntge_evolution = []
        
        # Initialize with baseline (best single model)
        if all_ind_NTGE is not None:
            # 如果有NTGE信息，优先选择NTGE最小的模型
            valid_ntge_indices = np.where(all_ind_NTGE != float('inf'))[0]
            if len(valid_ntge_indices) > 0:
                best_single_model = valid_ntge_indices[np.argmin(all_ind_NTGE[valid_ntge_indices])]
                current_ntge = all_ind_NTGE[best_single_model]
                print(f"🎯 选择NTGE最小的单模型 {best_single_model}: NTGE={current_ntge:.0f}, GE={all_ind_GE[best_single_model]:.4f}")
            else:
                best_single_model = np.argmin(all_ind_GE)
                current_ntge = float('inf')
                print(f"⚠️ 没有模型能达到GE=0，选择GE最小的单模型 {best_single_model}: GE={all_ind_GE[best_single_model]:.4f}")
        else:
            best_single_model = np.argmin(all_ind_GE)
            current_ntge = float('inf')
            print(f"📊 选择GE最小的单模型 {best_single_model}: GE={all_ind_GE[best_single_model]:.4f}")
        
        current_ge = all_ind_GE[best_single_model]
        
        total_reward = 0.0
        step = 0
        
        # Episode loop
        while step < max_steps and len(selected_models) < target_models and available_models:
            step_start_time = time.time()
            
            # Get current state (binary representation)
            current_state = self.get_binary_state(selected_models)
            
            # Select action (which model to add)
            if available_models:
                chosen_model = self.select_action_epsilon_greedy(
                    current_state, step, available_models, 
                    selected_models, is_training
                )
                
                # Update state
                prev_state = current_state.clone()
                prev_ge = current_ge
                prev_ntge = current_ntge
                selected_models.append(chosen_model)
                available_models.remove(chosen_model)
                new_state = self.get_binary_state(selected_models)
                
                # Evaluate new combination
                temp_ensemble = ensemble_predictions[selected_models]
                
                try:
                    ensemble_GE, _ = perform_attacks_ensemble(
                        nb_traces_attacks, temp_ensemble, plt_val, correct_key,
                        dataset=dataset, nb_attacks=nb_attacks, 
                        shuffle=True, leakage=leakage, byte=byte
                    )
                    new_ge = ensemble_GE[-1]
                    # 计算NTGE
                    new_ntge = NTGE_fn(ensemble_GE)
                    evaluation_successful = True
                except Exception as e:
                    print(f"⚠️ Step {step}: Attack evaluation failed: {str(e)}")
                    new_ge = current_ge * 1.1  # Penalty for failed evaluation
                    new_ntge = current_ntge  # 保持当前NTGE
                    evaluation_successful = False
                
                # Calculate reward with NTGE information
                reward = self.calculate_reward(prev_ge, new_ge, 
                                             selected_models[:-1], selected_models, 
                                             step, max_steps, prev_ntge, new_ntge)
                
                # Determine if episode should end
                done = (new_ge == 0.0 or 
                       len(selected_models) >= target_models or 
                       step >= max_steps - 1 or
                       not available_models)
                
                # Store transition in memory
                if is_training:
                    next_state = new_state if not done else None
                    self.memory.push(prev_state, chosen_model, next_state, reward, done)
                
                # Update for next iteration
                if evaluation_successful:
                    current_ge = new_ge
                    current_ntge = new_ntge
                    ge_evolution.append(new_ge)
                    ntge_evolution.append(new_ntge)
                else:
                    ge_evolution.append(current_ge)
                    ntge_evolution.append(current_ntge)
                
                total_reward += reward
                
                step_time = time.time() - step_start_time
                
                if is_training or step % 5 == 0:  # Reduce output in evaluation mode
                    ntge_str = f"{new_ntge:.0f}" if new_ntge != float('inf') else "∞"
                    print(f"  Step {step + 1}: Model {chosen_model} → GE {new_ge:.4f}, NTGE {ntge_str} "
                          f"(Reward: {reward:.3f}, Time: {step_time:.2f}s)")
                
                # Early stopping if perfect
                if new_ge == 0.0:
                    print(f"🎯 Perfect attack achieved at step {step + 1}!")
                    if new_ntge != float('inf'):
                        print(f"   NTGE: {new_ntge:.0f} 轨迹")
                    break
                    
            step += 1
        
        final_ge = ge_evolution[-1] if ge_evolution else current_ge
        final_ntge = ntge_evolution[-1] if ntge_evolution else current_ntge
        
        return {
            'selected_models': selected_models,
            'final_ge': final_ge,
            'final_ntge': final_ntge,
            'ge_evolution': ge_evolution,
            'ntge_evolution': ntge_evolution,
            'total_reward': total_reward,
            'episode_length': step,
            'converged': final_ge < 10.0
        }

    def find_minimum_traces_for_ge_zero(self, selected_models, ensemble_predictions,
                                       plt_attack, correct_key, dataset, leakage, 
                                       byte, perform_attacks_ensemble, max_traces=20000):
        """
        找到达到 GE=0 的最少攻击轨迹数
        使用二分搜索优化搜索过程
        """
        temp_ensemble = ensemble_predictions[selected_models]
        
        # 首先验证是否能在最大轨迹数内达到 GE=0
        logger.info(f"验证是否能在 {max_traces} 轨迹内达到 GE=0...")
        ensemble_GE, _ = perform_attacks_ensemble(
            max_traces, temp_ensemble, plt_attack, correct_key,
            dataset=dataset, nb_attacks=100, shuffle=True, 
            leakage=leakage, byte=byte
        )
        
        if ensemble_GE[-1] != 0:
            logger.warning(f"无法在 {max_traces} 轨迹内达到 GE=0，最终 GE={ensemble_GE[-1]}")
            return None, None, ensemble_GE
        
        logger.info(f"✅ 确认可以达到 GE=0，开始二分搜索最少轨迹数...")
        
        # 二分搜索找到最少轨迹数
        left, right = 1, max_traces
        best_traces = max_traces
        best_ge_curve = ensemble_GE
        
        while left <= right:
            mid = (left + right) // 2
            
            logger.info(f"测试 {mid} 轨迹...")
            ensemble_GE, _ = perform_attacks_ensemble(
                mid, temp_ensemble, plt_attack, correct_key,
                dataset=dataset, nb_attacks=50, shuffle=True,
                leakage=leakage, byte=byte
            )
            
            if ensemble_GE[-1] == 0:
                # 成功达到 GE=0，尝试更少的轨迹
                best_traces = mid
                best_ge_curve = ensemble_GE
                right = mid - 1
                logger.info(f"  ✓ GE=0 达成，继续搜索更少轨迹...")
            else:
                # 未达到 GE=0，需要更多轨迹
                left = mid + 1
                logger.info(f"  ✗ GE={ensemble_GE[-1]:.2f}，需要更多轨迹...")
        
        logger.info(f"🎯 最少攻击轨迹数: {best_traces}")
        
        # 计算 NTGE (Normalized Traces to Guessing Entropy)
        ntge = NTGE_fn(best_ge_curve)
        
        return best_traces, ntge, best_ge_curve
    
    def benchmark_and_select_optimal_traces(self, selected_models, ensemble_predictions,
                                          plt_attack, correct_key, dataset, leakage, 
                                          byte, perform_attacks_ensemble, max_traces=20000):
        """
        对选中的模型组合进行最优轨迹数搜索（以NTGE为核心指标）
        """
        if not selected_models:
            return None, None, None
            
        print(f"🔍 搜索最优轨迹数（基于NTGE）...")
        print(f"  选中模型: {selected_models}")
        print(f"  模型数量: {len(selected_models)}")
        
        temp_ensemble = ensemble_predictions[selected_models]
        trace_candidates = [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
        trace_candidates = [t for t in trace_candidates if t <= max_traces]
        
        optimal_traces = None
        optimal_ntge = float('inf')
        optimal_ge_curve = None
        best_ntge = float('inf')
        
        for nb_traces in trace_candidates:
            print(f"  测试 {nb_traces} 轨迹...")
            
            try:
                ensemble_GE, _ = perform_attacks_ensemble(
                    nb_traces, temp_ensemble, plt_attack, correct_key,
                    dataset=dataset, nb_attacks=30, shuffle=True, 
                    leakage=leakage, byte=byte
                )
                
                final_ge = ensemble_GE[-1]
                # 使用正确的NTGE函数计算
                ntge = NTGE_fn(ensemble_GE)
                
                print(f"    → GE: {final_ge:.3f}, NTGE: {ntge if ntge != float('inf') else '∞'}")
                
                # 优先选择NTGE最小的组合
                if ntge < best_ntge:
                    best_ntge = ntge
                    optimal_traces = nb_traces
                    optimal_ntge = ntge
                    optimal_ge_curve = ensemble_GE
                    print(f"    🎯 新的最优解: {nb_traces} 轨迹, NTGE: {ntge}")
                
                # 如果达到GE=0，可以提前停止（但继续检查是否有更小的NTGE）
                if final_ge == 0 and ntge != float('inf'):
                    print(f"    ✅ 达到GE=0，NTGE: {ntge}")
                    
            except Exception as e:
                print(f"    → 测试失败: {str(e)}")
                continue
        
        if optimal_traces is None:
            print(f"⚠️ 未找到有效的解决方案")
        else:
            print(f"🏆 最优解: {optimal_traces} 轨迹, NTGE: {optimal_ntge}")
        
        return optimal_traces, optimal_ntge, optimal_ge_curve

# Set up logger
logger = logging.getLogger(__name__) 