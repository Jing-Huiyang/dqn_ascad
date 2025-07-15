# 模型选择方法使用指南

本项目实现了三种不同的深度学习模型选择方法用于侧信道攻击集成学习。每种方法都有其特定的优势和适用场景。

## 三种选择方法概述

### 1. 原始递增方法 (Original Incremental)
**方法描述**: 使用DQN逐个选择模型，从空集开始，每次添加一个模型直到达到目标数量。

**特点**:
- ✅ 简单直观，易于理解
- ✅ 能保证选择数量的精确控制
- ❌ 可能陷入局部最优
- ❌ 无法进行模型替换优化

**适用场景**: 快速原型验证，计算资源有限的情况

### 2. 替换优化方法 (Replacement-based)
**方法描述**: 先选择初始的k个模型（基于个体GE排序），然后通过DQN学习不断尝试替换其中的模型来优化整体性能。

**特点**:
- ✅ 能跳出局部最优
- ✅ 考虑模型组合的全局优化
- ✅ 包含模拟退火机制避免过早收敛
- ❌ 计算开销较大
- ❌ 需要更多的超参数调整

**适用场景**: 追求最优性能，有充足计算资源的场景

### 3. 分层Top-K方法 (Hierarchical Top-K)
**方法描述**: 采用分层筛选策略，从粗选到精选逐步缩小候选范围，最后使用DQN进行精细选择。

**特点**:
- ✅ 计算效率高
- ✅ 结合了启发式筛选和智能选择
- ✅ 能处理大规模模型集合
- ❌ 可能在粗选阶段丢失好的组合
- ❌ 层数和每层候选数需要合理设置

**适用场景**: 大规模模型集合，需要平衡效率和效果的场景

## 使用方法

### 1. 修改 `select_models.py` 中的选择方法

```python
# 在 main() 函数中修改这个参数
selection_method = "replacement"  # 可选值: "original", "replacement", "hierarchical"
```

### 2. 运行单个方法

```bash
python select_models.py
```

### 3. 比较所有方法

```bash
python compare_selection_methods.py
```

## 方法参数配置

### 替换优化方法参数

```python
selected_models, ge_history = model_selector.select_models_replacement_based(
    # ... 其他参数 ...
    max_iterations=200,  # 最大优化迭代次数，建议范围: 100-300
)
```

- `max_iterations`: 控制优化的深度，值越大优化越充分但计算时间越长

### 分层Top-K方法参数

```python
selected_models, ge_history = model_selector.select_models_hierarchical_topk(
    # ... 其他参数 ...
    num_layers=3,  # 分层层数，建议2-4层
    candidates_per_layer=None  # 每层候选数量，None表示自动计算
)
```

- `num_layers`: 分层数量，通常3层效果较好
- `candidates_per_layer`: 可以手动指定，例如 `[50, 30, 20]`

## 性能分析

### 计算复杂度比较

| 方法 | 时间复杂度 | 空间复杂度 | 实际运行时间 |
|------|------------|------------|--------------|
| 原始递增 | O(k×n×t) | O(n) | 最快 |
| 替换优化 | O(m×n×t) | O(n) | 最慢 |
| 分层Top-K | O(L×n×t) | O(n) | 中等 |

其中: k=目标模型数量, n=总模型数量, t=攻击评估时间, m=替换迭代次数, L=分层评估次数

### 性能指标对比

运行 `compare_selection_methods.py` 后，会在 `./Result/ASCAD_mlp_byte2_HW/visualizations/` 目录下生成以下对比图表:

1. **method_comparison_summary.png**: 最终GE、NTGE和执行时间的对比
2. **method_comparison_curves.png**: 选择过程和攻击曲线的对比
3. **method_overlap_matrix.png**: 不同方法选择模型的重叠分析

## 最佳实践建议

### 1. 选择方法的决策树

```
开始
│
├─ 总模型数量 < 50?
│  ├─ 是 → 使用"原始递增"方法
│  └─ 否 ↓
│
├─ 计算资源充足且追求最优性能?
│  ├─ 是 → 使用"替换优化"方法
│  └─ 否 ↓
│
└─ 使用"分层Top-K"方法
```

### 2. 参数调优建议

**替换优化方法**:
- 如果GE改善停滞，增加 `max_iterations`
- 如果计算时间过长，减少 `max_iterations` 或降低 `nb_attacks`

**分层Top-K方法**:
- 对于100个模型，推荐层结构: `[60, 40, 20]`
- 对于500个模型，推荐层结构: `[200, 100, 50, 20]`

### 3. 结果解读

- **Final GE**: 越小越好，表示攻击效果越好
- **NTGE**: 越小越好，表示达到完全破解所需的轨迹数越少
- **Execution Time**: 在效果相近的情况下，优先选择时间更短的方法

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减少 `nb_attacks` 参数
   - 使用更小的 `batch_size`

2. **收敛过慢**
   - 调整DQN的学习率 `LR`
   - 修改探索率衰减参数 `EPS_DECAY`

3. **结果不稳定**
   - 增加 `nb_attacks` 以获得更稳定的评估
   - 运行多次取平均值

### 调试技巧

- 开启详细日志输出观察选择过程
- 使用小规模数据集进行快速测试
- 保存中间结果以便分析和调试

## 扩展功能

### 自定义选择策略

可以在 `ModelSelector` 类中添加新的选择方法:

```python
def select_models_custom(self, ensemble_predictions, all_ind_GE, ...):
    """
    自定义模型选择方法
    """
    # 实现你的选择逻辑
    pass
```

### 多目标优化

考虑同时优化多个指标:
- GE值 (攻击效果)
- 模型多样性
- 计算复杂度
- 存储需求

通过修改奖励函数可以实现多目标平衡。 