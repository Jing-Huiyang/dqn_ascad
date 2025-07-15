import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import MLP, CNN
from src.utils import perform_attacks, NTGE_fn
from tqdm import tqdm

def generate_predictions_from_existing_models():
    """
    从现有的训练模型中生成预测文件
    """
    # ==================== 参数设置 ====================
    dataset = "ASCAD_variable_desync50"
    model_type = "mlp"
    leakage = "HW"
    byte = 2
    nb_traces_attacks = 2000
    nb_attacks = 50
    
    # ==================== 路径设置 ====================
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_byte" + str(byte) + "_" + leakage + "/"
    model_root = save_root + "models/"
    
    print(f"Looking for models in: {model_root}")
    
    # ==================== 设备设置 ====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==================== 数据加载 ====================
    print("Loading datasets...")
    try:
        dset_test = Custom_Dataset(dataset=dataset, leakage=leakage,
                               transform=transforms.Compose([ToTensor_trace()]),
                               byte=byte, val_ratio=0.1)
        
        # 获取攻击数据 - 在choose_phase之前保存原始数据
        correct_key = dset_test.correct_key
        X_attack = dset_test.X_attack
        plt_attack = dset_test.plt_attack
        
        # 确保X_attack是numpy数组
        if not isinstance(X_attack, np.ndarray):
            raise ValueError(f"X_attack should be numpy array, got {type(X_attack)}")
        
        num_sample_pts = X_attack.shape[-1]
        
        print(f"Attack data shape: {X_attack.shape}")
        print(f"Correct key: {correct_key}")
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return
    
    # ==================== 设置类别数 ====================
    if leakage == 'HW':
        classes = 9
    elif leakage == 'ID':
        classes = 256
    else:
        print(f"Invalid leakage model: {leakage}")
        return
    
    # ==================== 查找现有模型 ====================
    model_files = []
    config_files = []
    
    for i in range(100):  # 假设最多100个模型
        model_path = model_root + f"model_{i}_byte{byte}.pth"
        config_path = model_root + f"model_configuration_{i}.npy"
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            model_files.append((i, model_path))
            config_files.append((i, config_path))
    
    print(f"Found {len(model_files)} trained models")
    
    if len(model_files) == 0:
        print("No trained models found! Please train models first.")
        return
    
    # ==================== 生成预测 ====================
    ensemble_predictions = []
    all_ind_GE = []
    
    print("Generating predictions from existing models...")
    
    for model_idx, model_path in tqdm(model_files, desc="Processing models"):
        try:
            # 加载模型配置
            config_path = model_root + f"model_configuration_{model_idx}.npy"
            config = np.load(config_path, allow_pickle=True).item()
            
            # 创建模型
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            else:
                print(f"Invalid model type: {model_type}")
                continue
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # 生成预测
            with torch.no_grad():
                attack_traces = torch.from_numpy(X_attack[:nb_traces_attacks]).to(device).unsqueeze(1).float()
                predictions_wo_softmax = model(attack_traces)
                predictions = F.softmax(predictions_wo_softmax, dim=1)
                predictions = predictions.cpu().numpy()
            
            ensemble_predictions.append(predictions)
            
            # 检查是否已有个体结果
            result_path = model_root + f"result_{model_idx}_byte{byte}.npy"
            
            if os.path.exists(result_path):
                # 加载现有结果
                individual_result = np.load(result_path, allow_pickle=True).item()
                individual_GE = individual_result["GE"]
                print(f"Model {model_idx}: Loaded existing GE = {individual_GE[-1]}")
            else:
                # 计算新的结果
                print(f"Model {model_idx}: Computing new GE...")
                individual_GE, _ = perform_attacks(
                    nb_traces_attacks, predictions, plt_attack, correct_key,
                    dataset=dataset, nb_attacks=nb_attacks, shuffle=True,
                    leakage=leakage, byte=byte
                )
                individual_NTGE = NTGE_fn(individual_GE)
                
                # 保存结果 - 使用allow_pickle=True来保存字典
                result_dict = {"GE": individual_GE, "NTGE": individual_NTGE}
                np.save(result_path, result_dict, allow_pickle=True)  # type: ignore
                print(f"Model {model_idx}: Computed GE = {individual_GE[-1]}")
            
            all_ind_GE.append(individual_GE[-1])
            
        except Exception as e:
            print(f"Error processing model {model_idx}: {str(e)}")
            continue
    
    # ==================== 保存集成预测文件 ====================
    if len(ensemble_predictions) > 0:
        print(f"\nSaving predictions for {len(ensemble_predictions)} models...")
        
        # 转换为numpy数组
        ensemble_predictions = np.array(ensemble_predictions)
        all_ind_GE = np.array(all_ind_GE)
        
        # 保存文件
        np.save(model_root + "all_ensemble_predictions.npy", ensemble_predictions)
        np.save(model_root + "all_ind_GE.npy", all_ind_GE)
        
        print("✅ Successfully saved:")
        print(f"   - all_ensemble_predictions.npy: shape {ensemble_predictions.shape}")
        print(f"   - all_ind_GE.npy: shape {all_ind_GE.shape}")
        print(f"   - GE values range: {all_ind_GE.min():.4f} to {all_ind_GE.max():.4f}")
        
        # 打印一些统计信息
        print(f"\n📊 Model Performance Statistics:")
        print(f"   - Best model GE: {all_ind_GE.min():.4f}")
        print(f"   - Worst model GE: {all_ind_GE.max():.4f}")
        print(f"   - Average GE: {all_ind_GE.mean():.4f}")
        print(f"   - Standard deviation: {all_ind_GE.std():.4f}")
        
        return True
    else:
        print("❌ No valid predictions generated!")
        return False

if __name__ == "__main__":
    print("🚀 Generating predictions from existing models...")
    success = generate_predictions_from_existing_models()
    
    if success:
        print("\n✅ Prediction generation completed successfully!")
        print("Now you can run the model selection methods:")
        print("   python select_models.py")
        print("   python compare_selection_methods.py")
    else:
        print("\n❌ Failed to generate predictions!") 