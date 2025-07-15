# 统一实验系统使用指南

## 🚀 系统概述

这个统一实验系统支持多数据集、多模型类型的深度学习侧信道攻击实验。系统采用配置管理模式，让您能够轻松切换不同的实验配置，并自动管理实验结果的存储和组织。

## 📁 目录结构

```
项目根目录/
├── config.py                    # 配置管理系统
├── train_models_unified.py      # 统一训练脚本
├── select_models_unified.py     # 统一模型选择脚本  
├── experiment_manager.py        # 实验管理器
├── src/                         # 核心模块
│   ├── dataloader.py           # 数据加载器
│   ├── net.py                  # 网络模型
│   ├── rl.py                   # 强化学习模型选择
│   ├── trainer.py              # 训练器
│   └── utils.py                # 工具函数
├── Dataset/                     # 数据集目录
│   ├── ASCAD/                  # ASCAD数据集
│   ├── ASCON/                  # ASCON数据集
│   └── ...                     # 其他数据集
└── Result/                      # 实验结果目录
    ├── ASCAD_mlp_byte2_HW/     # ASCAD + MLP + HW泄漏模型
    ├── ASCAD_cnn_byte2_HW/     # ASCAD + CNN + HW泄漏模型
    ├── ASCON_mlp_byte2_HW/     # ASCON + MLP + HW泄漏模型
    └── ...                     # 其他实验组合
```

## 🔧 支持的配置

### 📊 数据集
- **ASCAD**: 标准侧信道数据集，适合入门和对比实验
- **ASCAD_variable**: 可变密钥的ASCAD，更具挑战性
- **ASCAD_desync50/100**: 去同步化的ASCAD数据集
- **ASCON**: 轻量级加密算法数据集
- **AES_HD_ext**: 汉明距离模型数据集
- **CTF**: CTF竞赛数据集
- **ChipWhisperer**: 硬件实测数据集

### 🧠 模型类型
- **MLP**: 多层感知机，适合预处理数据，训练快速
- **CNN**: 卷积神经网络，适合原始轨迹，特征提取能力强

### 🔐 泄漏模型
- **HW**: 汉明重量模型 (9个类别: 0-8)
- **ID**: 身份模型 (256个类别: 0-255)

## 🎯 使用方法

### 1. 实验管理器（推荐方式）

#### 查看可用实验
```bash
python experiment_manager.py list
```

#### 运行单个实验
```bash
# 快速测试 (少量模型，快速验证)
python experiment_manager.py run --dataset ASCAD --model_type mlp --preset quick_test

# 完整实验 (100个模型，完整评估)  
python experiment_manager.py run --dataset ASCAD --model_type mlp --preset full_experiment

# 自定义配置
python experiment_manager.py run --dataset ASCAD --model_type cnn --mode full
```

#### 批量运行实验
```bash
# 对比不同模型类型
python experiment_manager.py batch --datasets ASCAD --model_types mlp cnn --preset quick_test

# 对比不同数据集
python experiment_manager.py batch --datasets ASCAD ASCON --model_types mlp --preset full_experiment

# 大规模对比实验
python experiment_manager.py batch --datasets ASCAD ASCON AES_HD_ext --model_types mlp cnn --preset quick_test
```

#### 检查实验状态
```bash
python experiment_manager.py status --dataset ASCAD --model_type mlp
```

#### 获取实验建议
```bash
python experiment_manager.py recommend
```

### 2. 直接使用脚本

#### 训练模型
```bash
# 基础训练
python train_models_unified.py --dataset ASCAD --model_type mlp

# 自定义参数
python train_models_unified.py --dataset ASCAD --model_type cnn --num_epochs 100 --total_num_models 50

# 使用预设配置
python train_models_unified.py --preset quick_test --dataset ASCON --model_type mlp
```

#### 模型选择
```bash
# 基础选择
python select_models_unified.py --dataset ASCAD --model_type mlp

# 自定义选择参数
python select_models_unified.py --dataset ASCAD --model_type mlp --num_top_k_model 15 --nb_attacks 30

# 使用预设配置
python select_models_unified.py --preset selection_only --dataset ASCAD --model_type cnn
```

### 3. 配置管理

#### 在Python中使用配置
```python
from config import get_config, ConfigTemplates, config_manager

# 获取默认配置
config = get_config(dataset="ASCAD", model_type="mlp")

# 使用预设配置模板
config = ConfigTemplates.quick_test_config("ASCAD", "mlp")

# 打印配置信息
config_manager.print_config(config)

# 创建实验目录
config_manager.create_directories("ASCAD", "mlp", 2, "HW")
```

## 📊 实验流程

### 典型实验流程
1. **选择数据集和模型类型**
2. **训练多个模型** (通常100个)
3. **使用强化学习选择最优模型组合** (通常选择20个)
4. **评估集成模型性能**
5. **分析和可视化结果**

### 推荐的实验策略

#### 初学者
```bash
# 1. 快速验证系统功能
python experiment_manager.py run --dataset ASCAD --model_type mlp --preset quick_test

# 2. 对比不同模型类型
python experiment_manager.py batch --datasets ASCAD --model_types mlp cnn --preset quick_test
```

#### 研究者
```bash
# 1. 完整单一实验
python experiment_manager.py run --dataset ASCAD --model_type mlp --preset full_experiment

# 2. 跨数据集泛化性测试
python experiment_manager.py batch --datasets ASCAD ASCAD_variable ASCON --model_types mlp --preset full_experiment

# 3. 模型架构对比研究
python experiment_manager.py batch --datasets ASCAD --model_types mlp cnn --preset full_experiment
```

## 📈 结果分析

### 结果文件结构
```
Result/ASCAD_mlp_byte2_HW/
├── models/                      # 模型文件
│   ├── model_0_byte2.pth       # 训练好的模型
│   ├── model_configuration_0.npy  # 模型超参数
│   ├── model_0_history.npy     # 训练历史
│   ├── ensemble_predictions.npy # 集成预测结果
│   └── ...
├── visualizations/              # 可视化文件
│   ├── ensemble_ge_curve.png   # 集成GE曲线
│   ├── ge_curve_model_0.png    # 个体模型GE曲线
│   └── ...
├── selection_results.npy        # 模型选择结果
└── training_summary.npy         # 训练总结
```

### 关键性能指标
- **GE (Guessing Entropy)**: 猜测熵，越低越好，0表示完全破解
- **NTGE**: 达到GE=0所需的轨迹数量
- **集成性能**: 多个模型组合后的性能
- **个体性能**: 单个模型的性能

### 结果加载示例
```python
import numpy as np

# 加载选择结果
results = np.load("Result/ASCAD_mlp_byte2_HW/selection_results.npy", allow_pickle=True).item()
print(f"集成GE: {results['ensemble_ge']}")
print(f"NTGE: {results['ntge']}")
print(f"选择的模型: {results['selected_indices']}")

# 加载集成预测
ensemble_pred = np.load("Result/ASCAD_mlp_byte2_HW/models/ensemble_predictions.npy")
```

## 🔧 自定义和扩展

### 添加新数据集
1. 在`src/utils.py`中添加数据加载函数
2. 在`src/dataloader.py`的`DatasetLoader.get_dataset_config()`中添加配置
3. 在`config.py`的`supported_datasets`中添加数据集名称

### 添加新模型类型
1. 在`src/net.py`中添加新的模型类
2. 在`ModelFactory.create_model()`中添加创建逻辑
3. 在`create_hyperparameter_space()`中添加超参数空间
4. 在`config.py`的`supported_model_types`中添加类型

### 自定义超参数
```python
from src.net import create_hyperparameter_space

# 获取默认超参数空间
search_space = create_hyperparameter_space("mlp")

# 自定义修改
search_space["layers"] = 5
search_space["neurons"] = 300
search_space["lr"] = 1e-4
```

## 🐛 故障排除

### 常见问题

#### 1. 数据集加载失败
```
❌ Error loading dataset: File not found
```
**解决方案**: 确保数据集文件存在于正确的路径下

#### 2. CUDA内存不足
```
❌ CUDA out of memory
```
**解决方案**: 减小batch size或使用CPU训练

#### 3. 模型选择失败
```
❌ No trained models found
```
**解决方案**: 确保先完成模型训练阶段

#### 4. 权限问题
```
❌ Permission denied when creating directory
```
**解决方案**: 检查目录写入权限

### 调试技巧

#### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 检查实验状态
```bash
python experiment_manager.py status --dataset ASCAD --model_type mlp
```

#### 验证配置
```python
from config import get_config
config = get_config(dataset="ASCAD", model_type="mlp")
print(config)
```

## 🚀 性能优化

### 训练优化
- 使用GPU加速训练
- 适当设置batch size
- 使用预设的quick_test进行初步验证

### 存储优化
- 定期清理不需要的模型文件
- 使用压缩存储大型预测文件

### 内存优化
- 批次处理大型数据集
- 及时释放不使用的模型

## 📚 扩展阅读

- [深度学习侧信道攻击论文集](papers/)
- [强化学习模型选择详解](RL_MODEL_SELECTION.md)
- [数据集详细说明](DATASET_GUIDE.md)
- [可视化分析指南](VISUALIZATION_GUIDE.md)

## 💬 获取帮助

如果您在使用过程中遇到问题，可以：

1. 查看此文档的故障排除部分
2. 运行`python experiment_manager.py recommend`获取建议
3. 检查日志文件了解详细错误信息
4. 使用`python experiment_manager.py status`检查实验状态

## 🎉 开始您的实验！

现在您已经了解了统一实验系统的使用方法，可以开始您的侧信道攻击研究了！

建议从快速测试开始：
```bash
python experiment_manager.py run --dataset ASCAD --model_type mlp --preset quick_test
```

祝您实验顺利！🎯 