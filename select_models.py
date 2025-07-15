import os
import numpy as np
import torch
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.rl import ModelSelector
from src.utils import perform_attacks_ensemble, NTGE_fn
import matplotlib.pyplot as plt
import seaborn as sns

def plot_selection_process(ge_history, save_path):
    """绘制模型选择过程中的GE变化"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ge_history, marker='o')
        plt.xlabel('Selection Step')
        plt.ylabel('Guessing Entropy')
        plt.title('GE Changes During Model Selection Process')
        plt.grid(True)
        plt.savefig(f"{save_path}/selection_process.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting selection process: {str(e)}")

def plot_selected_models_ge(selected_models, all_ge, save_path):
    """绘制被选中模型的GE分布"""
    try:
        plt.figure(figsize=(10, 6))
        selected_ge = [all_ge[i] for i in selected_models]
        sns.histplot(selected_ge, bins=10, label='Selected Models')
        plt.xlabel('Guessing Entropy')
        plt.ylabel('Number of Models')
        plt.title('GE Distribution of Selected Models')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/selected_models_ge.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting selected models GE: {str(e)}")

def plot_ensemble_ge_curve(ensemble_ge, save_path):
    """绘制集成攻击的GE曲线"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ensemble_ge, label='Ensemble Attack')
        plt.xlabel('Number of Traces')
        plt.ylabel('Guessing Entropy')
        plt.title('GE Curve for Ensemble Attack')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/ensemble_ge_curve.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting ensemble GE curve: {str(e)}")

def main():
    # ==================== 参数设置 ====================
    dataset = "ASCAD"         # 数据集选择：ASCAD/ASCON/ASCAD_variable
    model_type = "mlp"        # 模型类型：mlp(多层感知机)或cnn(卷积神经网络)
    leakage = "HW"           # 泄漏模型：HW(汉明重量)或ID(身份)
    byte = 2                 # 目标字节位置
    num_top_k_model = 20     # 最终选择的模型数量
    nb_traces_attacks = 2000 # 用于攻击的轨迹数量
    nb_attacks = 50         # 攻击次数
    
    # ==================== 新增：选择方法设置 ====================
    selection_method = "replacement"  # 选择方法：
                                      # "original" - 原始的逐个递增方法
                                      # "replacement" - 基于替换的优化方法
                                      # "hierarchical" - 分层top-k选择方法

    # ==================== 设置保存路径 ====================
    root = "jing/reinforce_learning_in_DLSCA_624/Result/ASCAD_mlp_byte2_HW/"
    save_root = root+dataset+"_"+model_type+ "_byte"+str(byte)+"_"+leakage+"/"
    model_root = save_root+"models/"

    # 检查目录是否存在
    try:
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        if not os.path.exists(model_root):
            os.makedirs(model_root, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return

    # ==================== 设备设置 ====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==================== 加载数据 ====================
    try:
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
        correct_key = dset_test.correct_key
        plt_attack = dset_test.plt_attack
    except Exception as e:
        print(f"Error getting attack data: {str(e)}")
        return

    # ==================== 加载模型预测结果 ====================
    print("Loading model predictions...")
    try:
        if not os.path.exists(model_root + "/all_ensemble_predictions.npy"):
            raise FileNotFoundError("Model predictions file not found. Please run train_models.py first.")
        if not os.path.exists(model_root + "/all_ind_GE.npy"):
            raise FileNotFoundError("Model GE values file not found. Please run train_models.py first.")
            
        ensemble_predictions = np.load(model_root + "/all_ensemble_predictions.npy", allow_pickle=True)
        all_ind_GE = np.load(model_root + "/all_ind_GE.npy", allow_pickle=True)
        print(f"Loaded predictions for {len(ensemble_predictions)} models")
        
        if len(ensemble_predictions) == 0:
            raise ValueError("No model predictions found in the loaded file.")
            
    except Exception as e:
        print(f"Error loading model predictions: {str(e)}")
        return

    # ==================== 使用选择的方法进行模型选择 ====================
    print(f"Using selection method: {selection_method}")
    try:
        model_selector = ModelSelector(device)
        
        if selection_method == "original":
            print("Starting original incremental model selection...")
            selected_models, ge_history = model_selector.select_models(
                ensemble_predictions=ensemble_predictions,
                all_ind_GE=all_ind_GE,
                num_top_k_model=num_top_k_model,
                perform_attacks_ensemble=perform_attacks_ensemble,
                nb_traces_attacks=nb_traces_attacks,
                plt_val=dset_val.plt_val,
                correct_key=correct_key,
                dataset=dataset,
                nb_attacks=nb_attacks,
                leakage=leakage
            )
            
        elif selection_method == "replacement":
            print("Starting replacement-based model selection...")
            selected_models, ge_history = model_selector.select_models_replacement_based(
                ensemble_predictions=ensemble_predictions,
                all_ind_GE=all_ind_GE,
                num_top_k_model=num_top_k_model,
                perform_attacks_ensemble=perform_attacks_ensemble,
                nb_traces_attacks=nb_traces_attacks,
                plt_val=dset_val.plt_val,
                correct_key=correct_key,
                dataset=dataset,
                nb_attacks=nb_attacks,
                leakage=leakage,
                max_iterations=200  # 最大迭代次数
            )
            
        elif selection_method == "hierarchical":
            print("Starting hierarchical top-k model selection...")
            selected_models, ge_history = model_selector.select_models_hierarchical_topk(
                ensemble_predictions=ensemble_predictions,
                all_ind_GE=all_ind_GE,
                num_top_k_model=num_top_k_model,
                perform_attacks_ensemble=perform_attacks_ensemble,
                nb_traces_attacks=nb_traces_attacks,
                plt_val=dset_val.plt_val,
                correct_key=correct_key,
                dataset=dataset,
                nb_attacks=nb_attacks,
                leakage=leakage,
                num_layers=3,  # 分层层数
                candidates_per_layer=None  # 自动计算每层候选数量
            )
            
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        if len(selected_models) == 0:
            raise ValueError("No models were selected by the selector.")
            
        print(f"Selected {len(selected_models)} models: {selected_models}")

        # 创建可视化保存目录
        vis_root = save_root + "visualizations/"
        if not os.path.exists(vis_root):
            os.makedirs(vis_root, exist_ok=True)

        # 绘制选择过程
        if ge_history:  # 确保有GE历史记录
            plot_selection_process(ge_history, vis_root)
            plot_selected_models_ge(selected_models, all_ind_GE, vis_root)
        
    except Exception as e:
        print(f"Error during model selection: {str(e)}")
        return

    # ==================== 执行集成攻击 ====================
    print("Performing ensemble attack with selected models...")
    try:
        ensemble_predictions = ensemble_predictions[selected_models]
        ensemble_GE, key_prob = perform_attacks_ensemble(nb_traces_attacks, ensemble_predictions, 
                                                       plt_attack, correct_key, dataset=dataset,
                                                       nb_attacks=nb_attacks, shuffle=True, 
                                                       leakage=leakage)
        ensemble_NTGE = NTGE_fn(ensemble_GE)

        # 绘制集成攻击的GE曲线
        if ensemble_GE is not None:  # 确保有GE值
            plot_ensemble_ge_curve(ensemble_GE, vis_root)
        
    except Exception as e:
        print(f"Error during ensemble attack: {str(e)}")
        return

    # ==================== 输出结果 ====================
    print("\nResults:")
    print("Selection method:", selection_method)
    print("Selected models:", selected_models)
    print("ensemble_GE", ensemble_GE)
    print("ensemble_NTGE", ensemble_NTGE)

    # ==================== 保存结果 ====================
    try:
        result_filename = f"result_ensemble_byte{byte}_NumModel_{num_top_k_model}_{selection_method}"
        np.save(model_root + result_filename, 
                {"GE": ensemble_GE, "NTGE": ensemble_NTGE, "selected_models": selected_models, 
                 "selection_method": selection_method, "ge_history": ge_history})
        print(f"Results saved to {model_root}/{result_filename}.npy")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main() 