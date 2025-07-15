import argparse
import os
import random
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
from src.utils import perform_attacks, NTGE_fn
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ge_curve(ge_values, model_idx, save_path):
    """绘制单个模型的GE曲线"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ge_values, label=f'Model {model_idx}')
        plt.xlabel('Number of Traces')
        plt.ylabel('Guessing Entropy')
        plt.title(f'GE Curve for Model {model_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/ge_curve_model_{model_idx}.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting GE curve for model {model_idx}: {str(e)}")

def plot_all_ge_distribution(all_ge, save_path):
    """绘制所有模型的GE值分布"""
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(all_ge, bins=20)
        plt.xlabel('Guessing Entropy')
        plt.ylabel('Number of Models')
        plt.title('Distribution of Final GE Values Across All Models')
        plt.grid(True)
        plt.savefig(f"{save_path}/all_ge_distribution.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting all GE distribution: {str(e)}")

def plot_training_curves(train_losses, val_losses, model_idx, save_path):
    """绘制训练过程中的损失曲线"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for Model {model_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/training_curve_model_{model_idx}.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting training curves for model {model_idx}: {str(e)}")

def main():
    # ==================== 参数设置 ====================
    train_models = True        # 是否训练新模型，False则加载已有模型
    dataset = "ASCAD"         # 数据集选择：ASCAD/ASCON/ASCAD_variable
    model_type = "mlp"        # 模型类型：mlp(多层感知机)或cnn(卷积神经网络)
    leakage = "HW"           # 泄漏模型：HW(汉明重量)或ID(身份)
    byte = 2                 # 目标字节位置
    num_epochs = 50          # 每个模型的训练轮数
    total_num_models = 100   # 要训练的总模型数量
    nb_traces_attacks = 2000 # 用于攻击的轨迹数量
    nb_attacks = 50         # 攻击次数

    # ==================== 目录创建 ====================
    try:
        if not os.path.exists('./Dataset/'):
            os.makedirs('./Dataset/', exist_ok=True)
        if not os.path.exists('./Result/'):
            os.makedirs('./Result/', exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return

    # ==================== 打印配置信息 ====================
    print(f"Leakage model: {leakage}")
    print(f"Dataset: {dataset}")
    print(f"model:{model_type} ")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"byte:{byte}")
    
    # ==================== 设置保存路径 ====================
    root = "./Result/"
    save_root = root+dataset+"_"+model_type+ "_byte"+str(byte)+"_"+leakage+"/"
    model_root = save_root+"models/"
    print("root:", root)
    print("save_time_path:", save_root)
    
    # 创建保存目录
    try:
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        if not os.path.exists(model_root):
            os.makedirs(model_root, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directories: {str(e)}")
        return

    # ==================== 设置随机种子 ====================
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==================== 设备设置 ====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==================== 攻击参数设置 ====================
    if leakage == 'HW':
        classes = 9   # 汉明重量泄漏模型的类别数
    elif leakage == 'ID':
        classes = 256 # 身份泄漏模型的类别数
    else:
        print(f"Invalid leakage model: {leakage}")
        return

    # ==================== 数据加载 ====================
    try:
        # 加载训练集
        dset_train = Custom_Dataset(dataset=dataset, leakage=leakage,
                                transform=transforms.Compose([ToTensor_trace()]),
                                byte=byte, val_ratio=0.1)
        dset_train.choose_phase("train")

        # 加载验证集
        dset_val = Custom_Dataset(dataset=dataset, leakage=leakage,
                              transform=transforms.Compose([ToTensor_trace()]),
                              byte=byte, val_ratio=0.1)
        dset_val.choose_phase("validation")

        # 加载测试集
        dset_test = Custom_Dataset(dataset=dataset, leakage=leakage,
                               transform=transforms.Compose([ToTensor_trace()]),
                               byte=byte, val_ratio=0.1)
        dset_test.choose_phase("test")
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return

    # ==================== 获取攻击数据 ====================
    try:
        correct_key = dset_train.correct_key
        X_attack = dset_test.X_attack
        Y_attack = dset_test.Y_attack
        plt_attack = dset_test.plt_attack
        num_sample_pts = X_attack.shape[-1]
    except Exception as e:
        print(f"Error getting attack data: {str(e)}")
        return

    # ==================== 模型训练和评估 ====================
    ensemble_predictions = []  # 存储所有模型的预测结果
    all_ind_GE = []  # 存储所有模型的GE值
    all_final_ge = []  # 存储所有模型的最终GE值

    # 创建可视化保存目录
    vis_root = save_root + "visualizations/"
    try:
        if not os.path.exists(vis_root):
            os.makedirs(vis_root, exist_ok=True)
    except Exception as e:
        print(f"Error creating visualization directory: {str(e)}")
        return

    # 训练或加载模型
    for num_models in range(total_num_models):
        print(f"\nTraining/Evaluating model {num_models + 1}/{total_num_models}")
        try:
            if train_models == True:
                model_path = model_root + f"model_{num_models}_byte{byte}.pth"
                if os.path.exists(model_path):
                    print(f"Model {num_models} already exists, skipping training.")
                    continue
                # 创建新的模型配置
                config = create_hyperparameter_space(model_type)
                np.save(model_root + "model_configuration_"+str(num_models)+".npy", config)
                
                # 设置数据加载器
                batch_size = config["batch_size"]
                num_workers = 2
                dataloaders = {
                    "train": torch.utils.data.DataLoader(dset_train, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers),
                    "val": torch.utils.data.DataLoader(dset_val, batch_size=batch_size,
                                                        shuffle=True, num_workers=num_workers)
                }
                dataset_sizes = {"train": len(dset_train), "val": len(dset_val)}
                print(f"训练集样本数: {len(dset_train)}, 验证集样本数: {len(dset_val)}")
                print(f"batch_size: {batch_size}")
                # 打印模型结构
                temp_model = MLP(config, num_sample_pts, classes) if model_type == "mlp" else CNN(config, num_sample_pts, classes)
                print("模型结构如下:")
                print(temp_model)
                # 训练模型
                start_time = time.time()
                model = trainer(config, num_epochs, num_sample_pts, dataloaders, 
                              dataset_sizes, model_type, classes, device,
                              model_root=model_root, model_idx=num_models)
                print(f"模型{num_models}训练用时: {time.time() - start_time:.2f}秒")
                # 获取训练过程中的损失并绘制曲线（在保存模型之前）
                if hasattr(model, 'train_losses') and hasattr(model, 'val_losses'):
                    train_losses = model.train_losses
                    val_losses = model.val_losses
                    print(f"train_losses: {train_losses}")
                    print(f"val_losses: {val_losses}")
                    if train_losses and val_losses:  # 确保有损失值
                        print(f"准备画图，保存到: {vis_root}/training_curve_model_{num_models}.png")
                        plot_training_curves(train_losses, val_losses, num_models, vis_root)
                    else:
                        print(f"Warning: Model {num_models} 没有有效的损失数据，无法画图")
                else:
                    print(f"Warning: Model {num_models} does not have train_losses or val_losses attributes")
                # 保存模型
                torch.save(model.state_dict(), model_path)
                print("Model saved.")
                # 检查每个epoch的batch数
                for phase in ['train', 'val']:
                    print(f"phase: {phase}, batch数: {len(dataloaders[phase])}")
                
            else:
                # 加载已有模型
                print("Loading existing model...")
                config = np.load(model_root + "model_configuration_"+str(num_models)+".npy", 
                               allow_pickle=True).item()
                if model_type == "mlp":
                    model = MLP(config, num_sample_pts, classes).to(device)
                elif model_type == "cnn":
                    model = CNN(config, num_sample_pts, classes).to(device)
                else:
                    print(f"Invalid model type: {model_type}")
                    continue
                
                try:
                    model.load_state_dict(torch.load(model_path))
                    
                    # 加载训练历史数据
                    history_path = model_root + "model_"+str(num_models)+"_history.npy"
                    print(f"Attempting to load history from: {history_path}")
                    if os.path.exists(history_path):
                        print("History file exists, loading...")
                        history = np.load(history_path, allow_pickle=True).item()
                        print("History loaded successfully")
                        model.train_losses = history['train_losses']
                        model.val_losses = history['val_losses']
                    else:
                        print(f"Warning: No training history found for model {num_models}")
                        model.train_losses = []
                        model.val_losses = []
                except Exception as e:
                    print(f"Error loading model {num_models}: {str(e)}")
                    print(f"Error type: {type(e)}")
                    continue
                
                print("Model loaded.")

            # 获取模型预测结果
            print("Getting model predictions...")
            attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
            predictions_wo_softmax = model(attack_traces)
            predictions = F.softmax(predictions_wo_softmax, dim=1)
            predictions = predictions.cpu().detach().numpy()
            ensemble_predictions.append(predictions)

            # 计算单个模型的性能
            print("Evaluating model performance...")
            save_individual_ge = True
            if save_individual_ge == True:
                individual_GE, individual_key_prob = perform_attacks(nb_traces_attacks, predictions, 
                                                                   plt_attack, correct_key, 
                                                                   dataset=dataset,
                                                                   nb_attacks=nb_attacks, 
                                                                   shuffle=True, 
                                                                   leakage=leakage, 
                                                                   byte=byte)
                individual_NTGE = NTGE_fn(individual_GE)
                np.save(model_root + "/result_"+str(num_models)+"_byte"+str(byte), 
                       {"GE": individual_GE, "NTGE": individual_NTGE})
            else:
                individual_result = np.load(model_root + "/result_"+str(num_models)+"_byte"+str(byte)+ ".npy", 
                                          allow_pickle=True).item()
                individual_GE = individual_result["GE"]
                individual_NTGE = individual_result["NTGE"]
            all_ind_GE.append(individual_GE[-1])
            print(f"Model {num_models + 1} evaluation completed. GE: {individual_GE[-1]}")
            
            # 绘制GE曲线
            if individual_GE is not None:  # 确保有GE值
                plot_ge_curve(individual_GE, num_models, vis_root)
            
            # 保存最终GE值
            all_final_ge.append(individual_GE[-1])
            
        except Exception as e:
            print(f"Error processing model {num_models + 1}: {str(e)}")
            continue

    # 绘制所有模型的GE分布
    if all_final_ge:  # 确保有GE值
        plot_all_ge_distribution(all_final_ge, vis_root)

    # 保存所有模型的预测结果和GE值
    print("\nSaving all model predictions and GE values...")
    try:
        np.save(model_root + "/all_ensemble_predictions.npy", ensemble_predictions)
        np.save(model_root + "/all_ind_GE.npy", all_ind_GE)
        print("All models have been trained and evaluated. Results saved.")
    except Exception as e:
        print(f"Error saving final results: {str(e)}")

if __name__ == "__main__":
    main() 