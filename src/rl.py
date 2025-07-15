import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple, deque
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, Callable
import numpy as np

# ==================== 经验回放相关定义 ====================
# 定义转换（transition）数据结构，用于存储经验
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    经验回放内存
    用于存储和采样(state, action, next_state, reward)转换
    """
    def __init__(self, capacity):
        """
        初始化经验回放内存
        Args:
            capacity: 内存容量
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        存储一个转换
        Args:
            *args: state, action, next_state, reward
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        随机采样一批转换
        Args:
            batch_size: 批量大小
        Returns:
            采样的转换列表
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        返回当前存储的转换数量
        """
        return len(self.memory)

# ==================== DQN网络定义 ====================
class DQN(nn.Module):
    """
    深度Q网络
    用于学习状态-动作值函数
    """
    def __init__(self, n_observations, n_actions):
        """
        初始化DQN网络
        Args:
            n_observations: 观察空间维度
            n_actions: 动作空间维度
        """
        super(DQN, self).__init__()
        # 三层全连接网络
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态
        Returns:
            Q值
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# ==================== 替换式DQN网络定义 ====================
class ReplacementDQN(nn.Module):
    """
    用于替换式选择的DQN网络
    输出每个模型的替换价值
    """
    def __init__(self, n_observations, total_models):
        super(ReplacementDQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, total_models)  # 输出每个模型的价值分数
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# ==================== 模型选择器定义 ====================
class ModelSelector:
    """
    模型选择器
    使用DQN学习最优的模型选择策略
    """
    def __init__(self, device, total_models=100):
        """
        初始化模型选择器
        Args:
            device: 计算设备（CPU/GPU）
            total_models: 总模型数量
        """
        self.device = device
        self.total_models = total_models
        self.n_actions = 2  # 0: not choose, 1: choose
        
        # 初始化观察空间
        self.observation_space = torch.zeros(total_models, device=device)
        
        # DQN超参数
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99      # 折扣因子
        self.EPS_START = 0.9   # 初始探索率
        self.EPS_END = 0.05    # 最小探索率
        self.EPS_DECAY = 1000  # 探索率衰减速度
        self.TAU = 0.005       # 目标网络更新率
        self.LR = 1e-4         # 学习率
        
        # 初始化网络
        self.policy_net = DQN(total_models, self.n_actions).to(device)
        self.target_net = DQN(total_models, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 初始化替换式网络
        self.replacement_policy_net = ReplacementDQN(total_models, total_models).to(device)
        self.replacement_target_net = ReplacementDQN(total_models, total_models).to(device)
        self.replacement_target_net.load_state_dict(self.replacement_policy_net.state_dict())
        
        # 初始化优化器和经验回放
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.replacement_optimizer = optim.AdamW(self.replacement_policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.replacement_memory = ReplayMemory(10000)
        self.steps_done = 0

    def reset_observation_space(self):
        """
        重置观察空间为全零向量
        Returns:
            重置后的观察空间
        """
        self.observation_space = torch.zeros(self.total_models, device=self.device)
        return self.observation_space

    def get_state(self, selected_models):
        """
        获取当前状态
        Args:
            selected_models: 已选择的模型索引列表
        Returns:
            当前状态向量
        """
        state = self.observation_space.clone()
        state[selected_models] = 1
        return state

    def select_action(self, state, current_step, is_training=True):
        """
        使用ε-贪婪策略选择动作
        Args:
            state: 当前状态
            current_step: 当前步骤
            is_training: 是否为训练模式 (True: 探索, False: 利用)
        Returns:
            action: 0或1
        """
        # 计算探索率
        if is_training:
            eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
        else:
            eps = 0.0 # 在评估模式下不探索
        
        # 将状态转换为张量
        state_tensor = state.unsqueeze(0)  # 添加批次维度
        
        # 计算Q值
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze()
        
        # ε-贪婪策略
        if random.random() < eps:
            # 探索：随机选择动作
            action = random.randrange(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            action = q_values.argmax().item()
        
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def select_replacement_models(self, state, selected_models, available_models):
        """
        选择要替换的模型对
        Args:
            state: 当前状态
            selected_models: 已选择的模型
            available_models: 可用的模型
        Returns:
            (model_to_remove, model_to_add): 要移除和添加的模型索引
        """
        if not selected_models or not available_models:
            return None, None
            
        state_tensor = state.unsqueeze(0)
        
        with torch.no_grad():
            model_values = self.replacement_policy_net(state_tensor).squeeze()
        
        # 选择已选模型中价值最低的（要移除）
        selected_values = model_values[selected_models]
        worst_selected_idx = selected_models[selected_values.argmin().item()]
        
        # 选择未选模型中价值最高的（要添加）
        available_values = model_values[available_models]
        best_available_idx = available_models[available_values.argmax().item()]
        
        return worst_selected_idx, best_available_idx

    def optimize_model(self):
        """
        优化DQN模型
        使用经验回放中的样本进行训练
        """
        if len(self.memory) < self.BATCH_SIZE:
            return
            
        # 从经验回放中采样
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 处理非终止状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        # 准备批次数据
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 计算当前Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # 计算损失并更新网络
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def optimize_replacement_model(self):
        """
        优化替换式DQN模型
        """
        if len(self.replacement_memory) < self.BATCH_SIZE:
            return
            
        transitions = self.replacement_memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        all_model_values = self.replacement_policy_net(state_batch)

        # 使用action_batch(包含被添加模型的索引)来获取相应动作的价值
        state_action_values = all_model_values.gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.replacement_target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1).detach())
        
        self.replacement_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.replacement_policy_net.parameters(), 100)
        self.replacement_optimizer.step()

    def update_target_networks(self):
        """更新目标网络"""
        # 软更新目标网络
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        # 更新替换网络
        replacement_target_net_state_dict = self.replacement_target_net.state_dict()
        replacement_policy_net_state_dict = self.replacement_policy_net.state_dict()
        for key in replacement_policy_net_state_dict:
            replacement_target_net_state_dict[key] = replacement_policy_net_state_dict[key]*self.TAU + replacement_target_net_state_dict[key]*(1-self.TAU)
        self.replacement_target_net.load_state_dict(replacement_target_net_state_dict)

    def select_models(self, ensemble_predictions, all_ind_GE, num_top_k_model, 
                    perform_attacks_ensemble, nb_traces_attacks, plt_val, 
                    correct_key, dataset, nb_attacks, leakage,
                    num_episodes=50, candidate_pool_size=50):
        """
        使用DQN通过多个回合进行训练和选择，以获得最优模型。
        Args:
            num_episodes: 训练的回合数。
            candidate_pool_size: 初始候选模型池的大小。
        Returns:
            best_model_selection: 找到的最佳模型组合。
            best_ge_history: 最佳回合的GE演变历史。
        """
        print(f"Starting DQN-based model selection with {num_episodes} training episodes...")
        
        # 1. 准备阶段：从所有模型中选出一个较小的候选池
        initial_candidates = np.argsort(all_ind_GE)[:candidate_pool_size].tolist()
        
        best_ge = float('inf')
        best_model_selection = []
        best_ge_history = []

        # 2. 训练阶段：运行多个回合以收集经验
        for i_episode in range(num_episodes):
            print(f"--- Episode {i_episode + 1}/{num_episodes} ---")
            
            # 运行一个探索性的选择回合
            _, ge_history, final_ge = self._run_selection_episode(
                ensemble_predictions, initial_candidates, num_top_k_model, all_ind_GE,
                perform_attacks_ensemble, nb_traces_attacks, plt_val, correct_key,
                dataset, nb_attacks, leakage, is_training=True
            )

            print(f"Episode {i_episode + 1} finished with final GE: {final_ge:.4f}")

            # 在每个回合后，进行模型优化
            if len(self.memory) > self.BATCH_SIZE:
                self.optimize_model()
            
            # 更新目标网络
            if i_episode % 10 == 0:
                self.update_target_networks()

        # 3. 决策阶段：关闭探索，使用学到的策略选择最终模型
        print("\n--- Final Selection Phase (Greedy Policy) ---")
        final_selection, final_ge_history, final_ge = self._run_selection_episode(
            ensemble_predictions, initial_candidates, num_top_k_model, all_ind_GE,
            perform_attacks_ensemble, nb_traces_attacks, plt_val, correct_key,
            dataset, nb_attacks, leakage, is_training=False
        )
        
        print(f"Final selected models: {final_selection}")
        print(f"Final GE: {final_ge:.4f}")
        
        return final_selection, final_ge_history

    def _run_selection_episode(self, ensemble_predictions, candidate_models, num_to_select, all_ind_GE, 
                             perform_attacks_ensemble, nb_traces_attacks, plt_val, 
                             correct_key, dataset, nb_attacks, leakage, is_training):
        """
        执行一个完整的模型选择回合。
        
        Args:
            is_training: 如果为True，则使用ε-贪婪进行探索并存储经验。
                         如果为False，则使用纯贪婪策略进行最终决策。
        Returns:
            (selected_models, ge_history, final_ge)
        """
        selected_models = []
        ge_history = []
        
        # 我们在一个固定的候选池中进行选择
        available_models = candidate_models.copy()
        
        # 状态表示当前已选择的模型 *在总模型列表中的位置*
        state = self.reset_observation_space() 
        current_ge = 256 # 初始GE设为最大值

        # 迭代选择，直到达到目标数量或没有可用模型
        for step in range(num_to_select):
            if not available_models:
                break
            
            # 1. 决定是否添加新模型 (Action)
            # 这里的状态应该反映选择"之前"的情况
            action = self.select_action(state, step, is_training=is_training)

            if action.item() == 1:  # 动作为1: 添加一个模型
                
                # 策略: 从可用模型中选择个体GE最低的那个
                best_available_model = min(available_models, key=lambda m_idx: all_ind_GE[m_idx])
                
                # 记录选择前的状态
                prev_state = state.clone()
                
                # 更新状态和模型列表
                selected_models.append(best_available_model)
                available_models.remove(best_available_model)
                state = self.get_state(selected_models) # 新状态

                # 2. 评估新组合并计算奖励 (Reward)
                temp_ensemble = ensemble_predictions[selected_models]
                ensemble_GE, _ = perform_attacks_ensemble(
                    nb_traces_attacks, temp_ensemble, plt_val, correct_key,
                    dataset=dataset, nb_attacks=nb_attacks, shuffle=True, leakage=leakage
                )
                new_ge = ensemble_GE[-1]
                ge_history.append(new_ge)

                # 奖励 = GE的降低值。我们希望奖励越大越好。
                reward = current_ge - new_ge
                current_ge = new_ge
                
                # 3. 存储经验 (如果正在训练)
                if is_training:
                    self.memory.push(prev_state.unsqueeze(0), action, state.unsqueeze(0), torch.tensor([reward], device=self.device))
            
            # 如果动作是0 (不添加)，则不发生任何事，进入下一步。
            # 这允许智能体 "跳过" 一步，如果它认为当前添加模型会使情况变糟。

        final_ge = ge_history[-1] if ge_history else current_ge
        return selected_models, ge_history, final_ge

    def select_models_replacement_based(self, ensemble_predictions, all_ind_GE, num_top_k_model, 
                                       perform_attacks_ensemble, nb_traces_attacks, plt_val, 
                                       correct_key, dataset, nb_attacks, leakage, max_iterations=200):
        """
        基于替换的模型选择方法
        Args:
            max_iterations: 最大优化迭代次数
        Returns:
            selected_models: 选中的模型索引列表
            ge_history: GE历史记录
        """
        print("Starting replacement-based model selection...")
        
        # 1. 初始选择：选择GE最小的num_top_k_model个模型
        initial_indices = np.argsort(all_ind_GE)[:num_top_k_model]
        selected_models = initial_indices.tolist()
        available_models = [i for i in range(len(ensemble_predictions)) if i not in selected_models]
        
        ge_history = []
        best_ge = float('inf')
        best_models = selected_models.copy()
        no_improvement_count = 0
        
        print(f"Initial selection: {selected_models}")
        
        # 计算初始性能
        temp_ensemble = ensemble_predictions[selected_models]
        initial_GE, _ = perform_attacks_ensemble(nb_traces_attacks, temp_ensemble, 
                                               plt_val, correct_key, dataset=dataset,
                                               nb_attacks=nb_attacks, shuffle=True, 
                                               leakage=leakage)
        current_ge = initial_GE[-1]
        best_ge = current_ge
        ge_history.append(current_ge)
        print(f"Initial ensemble GE: {current_ge}")
        
        # 2. 迭代优化过程
        for iteration in range(max_iterations):
            if not available_models:
                print("No more available models to try")
                break
                
            # 获取当前状态
            state = self.get_state(selected_models)
            
            # 选择要替换的模型对
            model_to_remove, model_to_add = self.select_replacement_models(
                state, selected_models, available_models)
            
            if model_to_remove is None or model_to_add is None:
                print("No valid replacement found")
                break
            
            # 执行替换
            new_selected_models = selected_models.copy()
            new_selected_models.remove(model_to_remove)
            new_selected_models.append(model_to_add)
            
            # 评估新的组合
            temp_ensemble = ensemble_predictions[new_selected_models]
            new_GE, _ = perform_attacks_ensemble(nb_traces_attacks, temp_ensemble, 
                                               plt_val, correct_key, dataset=dataset,
                                               nb_attacks=nb_attacks, shuffle=True, 
                                               leakage=leakage)
            new_ge = new_GE[-1]
            
            # 计算奖励
            reward = current_ge - new_ge  # 改进量作为奖励
            
            # 更新状态和经验
            new_state = self.get_state(new_selected_models)
            
            # 存储替换经验（使用被添加模型的索引作为动作）
            self.replacement_memory.push(
                state.unsqueeze(0), 
                torch.tensor([[model_to_add]], device=self.device, dtype=torch.long),
                new_state.unsqueeze(0), 
                torch.tensor([reward], device=self.device)
            )
            
            # 决定是否接受这次替换
            if new_ge < current_ge or (random.random() < 0.1):  # 10%概率接受更差的解（模拟退火）
                selected_models = new_selected_models
                available_models.remove(model_to_add)
                available_models.append(model_to_remove)
                current_ge = new_ge
                
                if new_ge < best_ge:
                    best_ge = new_ge
                    best_models = selected_models.copy()
                    no_improvement_count = 0
                    print(f"Iteration {iteration}: New best GE = {best_ge}, replaced {model_to_remove} with {model_to_add}")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            ge_history.append(current_ge)
            
            # 优化网络
            if iteration % 5 == 0:
                self.optimize_replacement_model()
                self.update_target_networks()
            
            # 早停条件
            if no_improvement_count >= 50:
                print(f"No improvement for 50 iterations, stopping early at iteration {iteration}")
                break
        
        print(f"Final selection: {best_models}, Best GE: {best_ge}")
        return best_models, ge_history

    def select_models_hierarchical_topk(self, ensemble_predictions, all_ind_GE, num_top_k_model, 
                                        perform_attacks_ensemble, nb_traces_attacks, plt_val, 
                                        correct_key, dataset, nb_attacks, leakage, 
                                        num_layers=3, candidates_per_layer=None):
        """
        分层top-k模型选择方法
        Args:
            num_layers: 分层层数
            candidates_per_layer: 每层候选模型数量，如果为None则自动计算
        Returns:
            selected_models: 选中的模型索引列表
            ge_history: GE历史记录
        """
        print("Starting hierarchical top-k model selection...")
        
        total_models = len(ensemble_predictions)
        if candidates_per_layer is None:
            # 自动计算每层的候选数量
            candidates_per_layer = [
                min(total_models, max(num_top_k_model * 8, 50)),  # 第一层：粗选
                min(total_models, max(num_top_k_model * 4, 30)),  # 第二层：精选
                num_top_k_model  # 第三层：最终选择
            ][:num_layers]
        
        current_candidates = list(range(total_models))
        ge_history = []
        
        print(f"Total models: {total_models}")
        print(f"Layer structure: {candidates_per_layer}")
        
        # 逐层选择
        for layer in range(num_layers):
            target_count = candidates_per_layer[layer]
            print(f"\nLayer {layer + 1}: Selecting {target_count} from {len(current_candidates)} candidates")
            
            if len(current_candidates) <= target_count:
                print(f"Current candidates already <= target, skipping layer {layer + 1}")
                continue
            
            if layer == 0:
                # 第一层：基于单模型GE值的快速筛选
                candidate_ge_values = [all_ind_GE[i] for i in current_candidates]
                sorted_indices = np.argsort(candidate_ge_values)
                current_candidates = [current_candidates[i] for i in sorted_indices[:target_count]]
                print(f"Layer {layer + 1}: Selected top {target_count} models by individual GE")
                
            elif layer == num_layers - 1:
                # 最后一层：使用DQN进行最终精选
                print(f"Layer {layer + 1}: Using DQN for final selection")
                selected_models, layer_ge_history = self._dqn_selection_from_candidates(
                    ensemble_predictions, current_candidates, target_count,
                    perform_attacks_ensemble, nb_traces_attacks, plt_val,
                    correct_key, dataset, nb_attacks, leakage, all_ind_GE
                )
                ge_history.extend(layer_ge_history)
                current_candidates = selected_models
                
            else:
                # 中间层：基于小批量集成性能的选择
                print(f"Layer {layer + 1}: Using ensemble-based selection")
                current_candidates = self._ensemble_based_selection(
                    ensemble_predictions, current_candidates, target_count,
                    perform_attacks_ensemble, nb_traces_attacks, plt_val,
                    correct_key, dataset, nb_attacks, leakage, all_ind_GE
                )
        
        print(f"Final hierarchical selection: {current_candidates}")
        return current_candidates, ge_history

    def _dqn_selection_from_candidates(self, ensemble_predictions, candidates, target_count,
                                      perform_attacks_ensemble, nb_traces_attacks, plt_val,
                                      correct_key, dataset, nb_attacks, leakage, all_ind_GE):
        """
        从候选模型中使用DQN选择指定数量的模型
        """
        # 创建映射：候选模型索引 -> 原始模型索引
        candidate_to_original = {i: candidates[i] for i in range(len(candidates))}
        
        # 重新构建候选模型的预测结果
        candidate_predictions = ensemble_predictions[candidates]
        candidate_ge_values = [all_ind_GE[i] for i in candidates]
        
        selected_candidates = []
        available_candidates = list(range(len(candidates)))
        ge_history = []
        
        # 重置观察空间（针对候选模型数量）
        original_total_models = self.total_models
        self.total_models = len(candidates)
        state = torch.zeros(len(candidates), device=self.device)
        
        try:
            while len(selected_candidates) < target_count and available_candidates:
                # 使用简化的贪心策略（因为候选集已经较小）
                if len(candidate_ge_values) > 0:
                    best_candidate_idx = available_candidates[np.argmin([candidate_ge_values[i] for i in available_candidates])]
                else:
                    best_candidate_idx = available_candidates[0]
                
                selected_candidates.append(best_candidate_idx)
                available_candidates.remove(best_candidate_idx)
                
                # 计算当前性能
                if len(selected_candidates) > 0:
                    temp_ensemble = candidate_predictions[selected_candidates]
                    ensemble_GE, _ = perform_attacks_ensemble(nb_traces_attacks, temp_ensemble,
                                                            plt_val, correct_key, dataset=dataset,
                                                            nb_attacks=nb_attacks, shuffle=True,
                                                            leakage=leakage)
                    ge_history.append(ensemble_GE[-1])
        
        finally:
            # 恢复原始总模型数量
            self.total_models = original_total_models
        
        # 将候选索引转换回原始模型索引
        selected_original_indices = [candidates[i] for i in selected_candidates]
        return selected_original_indices, ge_history

    def _ensemble_based_selection(self, ensemble_predictions, candidates, target_count,
                                 perform_attacks_ensemble, nb_traces_attacks, plt_val,
                                 correct_key, dataset, nb_attacks, leakage, all_ind_GE):
        """
        基于集成性能的模型选择
        """
        print(f"Evaluating {len(candidates)} candidates with ensemble-based selection")
        
        if len(candidates) <= target_count:
            return candidates
        
        # 使用贪心算法：逐个添加能最大改善集成性能的模型
        selected_models = []
        remaining_candidates = candidates.copy()
        
        # 随机选择一个初始模型
        if remaining_candidates:
            initial_idx = remaining_candidates[np.argmin([all_ind_GE[i] for i in remaining_candidates])]
            selected_models.append(initial_idx)
            remaining_candidates.remove(initial_idx)
        
        # 贪心添加剩余模型
        while len(selected_models) < target_count and remaining_candidates:
            best_candidate = None
            best_ge = float('inf')
            
            # 评估每个剩余候选模型
            for candidate in remaining_candidates[:min(20, len(remaining_candidates))]:  # 限制评估数量以提高效率
                test_models = selected_models + [candidate]
                temp_ensemble = ensemble_predictions[test_models]
                
                try:
                    ensemble_GE, _ = perform_attacks_ensemble(
                        min(nb_traces_attacks, 1000),  # 使用较少的轨迹数以提高速度
                        temp_ensemble, plt_val, correct_key,
                        dataset=dataset, nb_attacks=min(nb_attacks, 10),
                        shuffle=True, leakage=leakage
                    )
                    current_ge = ensemble_GE[-1]
                    
                    if current_ge < best_ge:
                        best_ge = current_ge
                        best_candidate = candidate
                        
                except Exception as e:
                    print(f"Error evaluating candidate {candidate}: {e}")
                    continue
            
            if best_candidate is not None:
                selected_models.append(best_candidate)
                remaining_candidates.remove(best_candidate)
                print(f"Added model {best_candidate}, current ensemble size: {len(selected_models)}, GE: {best_ge}")
            else:
                # 如果没有找到更好的候选者，随机选择一个
                if remaining_candidates:
                    fallback_candidate = remaining_candidates[0]
                    selected_models.append(fallback_candidate)
                    remaining_candidates.remove(fallback_candidate)
                    print(f"Fallback: added model {fallback_candidate}")
        
        return selected_models