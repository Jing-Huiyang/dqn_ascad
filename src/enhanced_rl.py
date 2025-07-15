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
        global_features = self.global_context(x.mean(dim=0, keepdim=True) if len(x.shape) > 1 else x)
        
        # Combine local and global features
        combined_features = torch.cat([x, global_features.expand_as(x[:x.shape[0]//1])], dim=-1)
        
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
            
        return action
        
    def calculate_reward(self, prev_ge: float, new_ge: float, 
                        prev_models: List[int], new_models: List[int],
                        episode_step: int) -> float:
        """
        计算奖励函数（考虑多个因素）
        """
        # Primary reward: GE improvement
        ge_improvement = prev_ge - new_ge
        
        # Model count penalty (encourage efficiency)
        model_count_penalty = len(new_models) * 0.01
        
        # Early success bonus
        early_success_bonus = 0.0
        if new_ge == 0:
            early_success_bonus = max(0, (self.max_models - len(new_models))) * 0.1
            
        # Step penalty (encourage faster convergence)
        step_penalty = episode_step * 0.001
        
        # Combined reward
        reward = ge_improvement + early_success_bonus - model_count_penalty - step_penalty
        
        return reward
        
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
                                       target_models, perform_attacks_ensemble, 
                                       nb_traces_attacks, plt_val, correct_key, 
                                       dataset, nb_attacks, leakage, byte,
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
        best_combination = []
        best_ge = float('inf')
        best_ge_evolution = []
        
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
                episode, max_episode_length, is_training=True
            )
            
            # Update best results
            if episode_results['final_ge'] < best_ge:
                best_ge = episode_results['final_ge']
                best_combination = episode_results['selected_models'].copy()
                best_ge_evolution = episode_results['ge_evolution'].copy()
                
                print(f"🎉 New best combination found!")
                print(f"   Models: {best_combination}")
                print(f"   Final GE: {best_ge:.4f}")
                print(f"   Model count: {len(best_combination)}")
            
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
            num_episodes, max_episode_length, is_training=False
        )
        
        if final_results['final_ge'] < best_ge:
            best_ge = final_results['final_ge']
            best_combination = final_results['selected_models'].copy()
            best_ge_evolution = final_results['ge_evolution'].copy()
            print(f"🎉 Final greedy policy achieved new best!")
        
        print(f"\n🏆 最佳结果:")
        print(f"   选择模型: {best_combination}")
        print(f"   模型数量: {len(best_combination)}")
        print(f"   最终GE: {best_ge:.4f}")
        print(f"   收敛性: {'良好' if best_ge < 10 else '需改进'}")
        
        return best_combination, best_ge_evolution
        
    def _run_global_strategy_episode(self, ensemble_predictions, all_ind_GE, target_models,
                                   perform_attacks_ensemble, nb_traces_attacks, plt_val,
                                   correct_key, dataset, nb_attacks, leakage, byte,
                                   episode, max_steps, is_training=True):
        """
        运行单个全局策略回合
        """
        
        selected_models = []
        available_models = list(range(self.total_models))
        ge_evolution = []
        
        # Initialize with baseline (best single model)
        best_single_model = np.argmin(all_ind_GE)
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
                    evaluation_successful = True
                except Exception as e:
                    print(f"⚠️ Step {step}: Attack evaluation failed: {str(e)}")
                    new_ge = current_ge * 1.1  # Penalty for failed evaluation
                    evaluation_successful = False
                
                # Calculate reward
                reward = self.calculate_reward(current_ge, new_ge, 
                                             selected_models[:-1], selected_models, step)
                
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
                    ge_evolution.append(new_ge)
                else:
                    ge_evolution.append(current_ge)
                
                total_reward += reward
                
                step_time = time.time() - step_start_time
                
                if is_training or step % 5 == 0:  # Reduce output in evaluation mode
                    print(f"  Step {step + 1}: Model {chosen_model} → GE {new_ge:.4f} "
                          f"(Reward: {reward:.3f}, Time: {step_time:.2f}s)")
                
                # Early stopping if perfect
                if new_ge == 0.0:
                    print(f"🎯 Perfect attack achieved at step {step + 1}!")
                    break
                    
            step += 1
        
        final_ge = ge_evolution[-1] if ge_evolution else current_ge
        
        return {
            'selected_models': selected_models,
            'final_ge': final_ge,
            'ge_evolution': ge_evolution,
            'total_reward': total_reward,
            'episode_length': step,
            'converged': final_ge < 10.0
        }

    def benchmark_and_select_optimal_traces(self, selected_models, ensemble_predictions,
                                          plt_attack, correct_key, dataset, leakage, 
                                          byte, perform_attacks_ensemble, max_traces=20000):
        """
        对选中的模型组合进行最优轨迹数搜索
        """
        if not selected_models:
            return None, None, None
            
        print(f"🔍 搜索最优轨迹数...")
        print(f"  选中模型: {selected_models}")
        print(f"  模型数量: {len(selected_models)}")
        
        temp_ensemble = ensemble_predictions[selected_models]
        trace_candidates = [500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
        trace_candidates = [t for t in trace_candidates if t <= max_traces]
        
        optimal_traces = None
        optimal_ntge = None
        optimal_ge_curve = None
        
        for nb_traces in trace_candidates:
            print(f"  测试 {nb_traces} 轨迹...")
            
            try:
                ensemble_GE, _ = perform_attacks_ensemble(
                    nb_traces, temp_ensemble, plt_attack, correct_key,
                    dataset=dataset, nb_attacks=30, shuffle=True, 
                    leakage=leakage, byte=byte
                )
                
                final_ge = ensemble_GE[-1]
                ntge = np.sum(ensemble_GE)  # Simple NTGE calculation
                
                print(f"    → GE: {final_ge:.3f}, NTGE: {ntge:.1f}")
                
                if final_ge == 0:
                    print(f"🎉 找到最优解: {nb_traces} 轨迹")
                    optimal_traces = nb_traces
                    optimal_ntge = ntge
                    optimal_ge_curve = ensemble_GE
                    break
                    
            except Exception as e:
                print(f"    → 测试失败: {str(e)}")
                continue
        
        if optimal_traces is None:
            print(f"⚠️ 未找到GE=0的解决方案")
        
        return optimal_traces, optimal_ntge, optimal_ge_curve

# Set up logger
logger = logging.getLogger(__name__) 