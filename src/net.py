import math
import random
from typing import Dict, Any, Union

import torch.nn as nn
import torch.nn.functional as F


class ModelFactory:
    """
    模型工厂类，统一管理不同类型模型的创建
    """
    
    @staticmethod
    def create_model(model_type: str, search_space: Dict[str, Any], 
                    num_sample_pts: int, classes: int) -> nn.Module:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型 ("mlp" 或 "cnn")
            search_space: 超参数搜索空间
            num_sample_pts: 输入特征维度
            classes: 输出类别数
            
        Returns:
            创建的模型实例
        """
        model_type = model_type.lower()
        
        if model_type == "mlp":
            return MLP(search_space, num_sample_pts, classes)
        elif model_type == "cnn":
            return CNN(search_space, num_sample_pts, classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: ['mlp', 'cnn']")
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        获取模型类型的信息
        """
        model_info = {
            "mlp": {
                "name": "Multi-Layer Perceptron",
                "description": "全连接神经网络，适用于表格数据和特征工程后的数据",
                "advantages": ["训练速度快", "参数较少", "可解释性较好"],
                "recommended_for": ["ASCAD", "AES_HD_ext", "CTF"]
            },
            "cnn": {
                "name": "Convolutional Neural Network", 
                "description": "卷积神经网络，适用于原始时间序列数据",
                "advantages": ["特征提取能力强", "适应性好", "处理原始波形效果好"],
                "recommended_for": ["ChipWhisperer", "ASCON", "原始轨迹数据"]
            }
        }
        
        model_type = model_type.lower()
        if model_type not in model_info:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model_info[model_type]


class MLP(nn.Module):
    def __init__(self, search_space, num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]

        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(num_sample_pts, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x)  # F.softmax()
        x = x.squeeze(1)
        return x


class CNN(nn.Module):
    def __init__(self, search_space, num_sample_pts, classes):
        super(CNN, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]
        self.conv_layers = search_space["conv_layers"]

        self.layers = nn.ModuleList()
        # CNN
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(
            search_space)
        num_features = num_sample_pts
        for layer_index in range(0, self.conv_layers):
            # Convolution layer
            new_out_channels = self.filters[layer_index]
            if layer_index == 0:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=self.kernels[layer_index],
                                                           stride=self.kernels[layer_index],
                                                           padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))

            else:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=self.kernels[layer_index],
                                                           stride=self.kernels[layer_index],
                                                           padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(
                    nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                              stride=conv1d_stride, padding=self.paddings[layer_index]))
            # Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            # Pooling Layer
            if self.pooling_type[layer_index] == "max_pool":
                layer_pool_size = self.pooling_sizes[layer_index]
                layer_pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_maxpool1d(num_features, layer_pool_size, layer_pool_stride)

                if new_num_features <= 0:
                    layer_pool_size = 1
                    layer_pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=layer_pool_size, stride=layer_pool_stride))
            elif self.pooling_type[layer_index] == "average_pool":
                pool_size = self.pooling_sizes[layer_index]
                pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_avgpool1d(num_features, pool_size, pool_stride)
                if new_num_features <= 0:
                    pool_size = 1
                    pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            # BatchNorm
            self.layers.append(nn.BatchNorm1d(new_out_channels))
            prev_out_channels = new_out_channels
        # MLP
        self.layers.append(nn.Flatten())
        # Flatten
        flatten_neurons = prev_out_channels * num_features
        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(flatten_neurons, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))
            # Activation layer
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x)  # F.softmax()
        x = x.squeeze(1)
        return x


def cal_num_features_conv1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out


def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out


def cal_num_features_avgpool1d(n_sample_points, kernel_size, stride, padding=0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"]  # size == stride
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"]
    init_kernels = search_space["kernels"]  # stride = kernel/2
    init_padding = search_space["padding"]  # only for conv1d layers.
    kernels = []
    strides = []
    filters = []
    paddings = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    for conv_layers_index in range(1, conv_layers + 1):
        if conv_layers_index == 1:
            filters.append(init_filters)
            kernels.append(init_kernels)
            strides.append(int(init_kernels / 2))
            paddings.append(init_padding)
        else:
            filters.append(filters[conv_layers_index - 2] * 2)
            kernels.append(kernels[conv_layers_index - 2] // 2)
            strides.append(int(kernels[conv_layers_index - 2] // 4))
            paddings.append(init_padding)
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)
    return kernels, strides, filters, pooling_type, pooling_sizes, pooling_strides, paddings


def weight_init(m, type='kaiming_uniform_'):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('selu'))
        elif type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            nn.init.uniform_(m.weight)
        if m.bias != None:
            nn.init.zeros_(m.bias)


def create_hyperparameter_space(model_type):
    """
    创建模型类型对应的超参数搜索空间
    
    Args:
        model_type: 模型类型 ("mlp" 或 "cnn")
        
    Returns:
        超参数字典
    """
    model_type = model_type.lower()
    
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                        "lr": random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                        "optimizer": random.choice(["RMSprop", "Adam"]),
                        "layers": random.randrange(1, 8, 1),
                        "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
                        "activation": random.choice(["relu", "selu", "elu", "tanh"]),
                        "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                        }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                        "lr": random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                        "optimizer": random.choice(["RMSprop", "Adam"]),
                        "layers": random.randrange(1, 8, 1),
                        "neurons": random.choice([10, 20, 50, 100, 200, 300, 400, 500]),
                        "activation": random.choice(["relu", "selu", "elu", "tanh"]),
                        "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                        "pooling_types": random.choice(["max_pool", "average_pool"]),
                        "pooling_sizes": random.choice([2, 4, 6, 8, 10]),  # size == strides
                        "conv_layers": random.choice([1, 2, 3, 4]),
                        "filters": random.choice([4, 8, 12, 16]),
                        "kernels": random.choice([i for i in range(26, 53, 2)]),  # strides = kernel/2
                        "padding": random.choice([0, 4, 8, 12, 16]),
                        }
        return search_space
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: ['mlp', 'cnn']")


def print_model_comparison():
    """
    打印不同模型类型的对比信息
    """
    print("\n🧠 Model Type Comparison:")
    print("=" * 60)
    
    for model_type in ["mlp", "cnn"]:
        info = ModelFactory.get_model_info(model_type)
        print(f"\n📊 {model_type.upper()} - {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Advantages: {', '.join(info['advantages'])}")
        print(f"   Recommended for: {', '.join(info['recommended_for'])}")


if __name__ == "__main__":
    # 测试模型工厂
    print_model_comparison()

