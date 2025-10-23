#!/usr/bin/env python3
"""
🚀 ULTIMATE PIPELINE for DNA Binding Site Prediction

整合所有优化:
1. 增强版条件扩散模型 (EnhancedConditionalDiffusionModel)
2. 高级GAT-GNN模型 (AdvancedBindingSiteGNN)  
3. PPI边预测器
4. 自适应数据增强
5. 交叉验证 + 集成学习

预期性能提升:
- 数据质量: 0.178 → 0.65+
- 数据比例: 22% → 90%
- F1 Score: 0.48 → 0.60+
- MCC: 0.42 → 0.55+
"""
import os
import sys
import time
import glob
import json
import warnings
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch_geometric.data import Data

warnings.filterwarnings('ignore')

# 导入配置
from ultimate_config import UltimateConfig
from data_loader import ProteinDataset
from main import calculate_class_ratio,set_seed

# 导入增强版模型
try:
    from enhanced_diffusion_model import EnhancedConditionalDiffusionModel
    ENHANCED_DIFFUSION_AVAILABLE = True
    print("✅ 增强版扩散模型已加载")
except ImportError as e:
    print(f"⚠️  增强版扩散模型不可用: {e}")
    print("   使用标准扩散模型")
    from ddpm_diffusion_model import EnhancedDiffusionModel
    ENHANCED_DIFFUSION_AVAILABLE = False

try:
    from advanced_gnn_model import AdvancedBindingSiteGNN
    ADVANCED_GNN_AVAILABLE = True
    print("✅ 高级GNN模型已加载")
except ImportError as e:
    print(f"⚠️  高级GNN模型不可用: {e}")
    print("   使用标准GNN模型")
    from improved_gnn_model import ImprovedBindingSiteGNN
    ADVANCED_GNN_AVAILABLE = False

# 导入增强模块
from ultimate_augmentation import ultimate_augment_dataset
from edge_predictor_augmentation import ImprovedEdgePredictor


def load_dataset_quiet(dataset_file, config):
    """静默加载数据集"""
    import shutil
    import contextlib

    temp_data_dir = os.path.join(config.data_dir, "temp_ultimate")
    os.makedirs(temp_data_dir, exist_ok=True)

    temp_file = os.path.join(temp_data_dir, os.path.basename(dataset_file))
    shutil.copy2(dataset_file, temp_file)

    try:
        if not config.verbose_loading:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    dataset_loader = ProteinDataset(temp_data_dir, device=config.device)
                    dataset = dataset_loader.proteins
        else:
            dataset_loader = ProteinDataset(temp_data_dir, device=config.device)
            dataset = dataset_loader.proteins
    finally:
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)

    return dataset


def load_edge_predictor(config):
    """加载边预测器"""
    print(f"\n🔗 加载边预测器 (超稳定版)...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ppi_model_path = os.path.join(current_dir, "models", "edge_predictor_best_ultra_stable.pth")
    
    if not os.path.exists(ppi_model_path):
        print(f"❌ 边预测器模型不存在: {ppi_model_path}")
        return None
    
    # 加载checkpoint
    checkpoint = torch.load(ppi_model_path, map_location='cpu', weights_only=False)
    
    # 推断hidden_dim
    if 'model_state_dict' in checkpoint:
        hidden_dim = checkpoint['model_state_dict']['fc_transform.weight'].shape[0]
    else:
        hidden_dim = checkpoint['fc_transform.weight'].shape[0]
    
    # 创建边预测器
    edge_predictor = ImprovedEdgePredictor(
        input_dim=1280,  # ESM2特征维度
        hidden_dim=hidden_dim
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        edge_predictor.load_state_dict(checkpoint['model_state_dict'])
    else:
        edge_predictor.load_state_dict(checkpoint)
    
    edge_predictor.to(config.device)
    edge_predictor.eval()
    
    print(f"✅ 边预测器已加载 (hidden_dim={hidden_dim})")
    print(f"   训练AUC: {checkpoint.get('best_auc', 'N/A')}")
    
    return edge_predictor


def train_enhanced_diffusion_model(train_dataset, config):
    """训练增强版扩散模型"""
    print(f"\n🧠 训练增强版扩散模型...")
    
    if ENHANCED_DIFFUSION_AVAILABLE and config.use_enhanced_diffusion:
        # 使用条件扩散模型
        diffusion_model = EnhancedConditionalDiffusionModel(
            input_dim=config.diffusion_input_dim,
            T=config.enhanced_diffusion_config['T'],
            hidden_dim=config.enhanced_diffusion_config['hidden_dim'],
            context_dim=config.enhanced_diffusion_config['context_dim'],
            device=config.device
        )
        
        print(f"  模型类型: 条件扩散 (Conditional DDPM)")
        print(f"  T={config.enhanced_diffusion_config['T']}, context_dim={config.enhanced_diffusion_config['context_dim']}")
    else:
        # 使用标准扩散模型
        from ddpm_diffusion_model import EnhancedDiffusionModel
        diffusion_model = EnhancedDiffusionModel(
            input_dim=config.diffusion_input_dim,
            T=config.diffusion_T,
            device=config.device
        )
        print(f"  模型类型: 标准扩散 (DDPM)")
    
    # 训练
    start_time = time.time()
    diffusion_model.train_on_positive_samples(
        train_dataset,
        epochs=config.diffusion_epochs,
        batch_size=config.diffusion_batch_size,
        lr=config.diffusion_lr
    )
    train_time = time.time() - start_time
    
    print(f"✅ 扩散模型训练完成: {train_time:.1f}秒")
    
    return diffusion_model


def create_gnn_model(config):
    """创建GNN模型"""
    if ADVANCED_GNN_AVAILABLE and config.use_advanced_gnn:
        # 使用高级GAT-GNN
        model = AdvancedBindingSiteGNN(
            input_dim=config.diffusion_input_dim,
            hidden_dim=config.advanced_gnn_config['hidden_dim'],
            num_layers=config.advanced_gnn_config['num_layers'],
            heads=config.advanced_gnn_config['heads'],
            dropout=config.advanced_gnn_config['dropout'],
            use_edge_features=config.advanced_gnn_config['use_edge_features'],
            focal_alpha=config.advanced_gnn_config['focal_alpha'],
            focal_gamma=config.advanced_gnn_config['focal_gamma'],
            class_balanced=config.advanced_gnn_config['class_balanced']
        )
        print(f"  模型类型: Advanced GAT-GNN")
        print(f"  层数={config.advanced_gnn_config['num_layers']}, 头数={config.advanced_gnn_config['heads']}")
    else:
        # 使用标准GNN
        from improved_gnn_model import ImprovedBindingSiteGNN
        model = ImprovedBindingSiteGNN(
            input_dim=config.diffusion_input_dim,
            hidden_dim=config.gnn_hidden_dim,
            dropout=config.gnn_dropout
        )
        print(f"  模型类型: 标准GNN")
    
    return model


def cross_validation_training(augmented_data, original_data, config):
    """交叉验证训练"""
    print(f"\n🔄 {config.cv_folds}折交叉验证训练...")
    
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(original_data)):
        print(f"\n📊 第 {fold+1}/{config.cv_folds} 折")
        
        # 准备数据
        val_original_indices = set(val_idx)
        train_original = [original_data[i] for i in range(len(original_data))
                         if i not in val_original_indices]
        train_fold = augmented_data + train_original
        val_fold = [original_data[i] for i in val_idx]
        
        print(f"  训练集: {len(train_fold)} (增强: {len(augmented_data)}, 原始: {len(train_original)})")
        print(f"  验证集: {len(val_fold)}")
        
        # 创建模型
        model = create_gnn_model(config)
        
        # 训练
        if ADVANCED_GNN_AVAILABLE and config.use_advanced_gnn:
            # 使用高级GNN的训练方法
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(train_fold, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_fold, batch_size=1, shuffle=False)
            
            best_f1 = model.train_model(
                train_loader,
                val_loader,
                epochs=config.gnn_epochs,
                lr=config.gnn_lr,
                device=config.device,
                patience=config.gnn_patience
            )
            
            # 评估
            metrics = model.evaluate(val_loader, config.device)
            best_auc = metrics['auc_pr']
        else:
            # 使用标准GNN的训练方法
            model.to(config.device)
            best_auc, best_f1 = model.train_with_early_stopping(
                train_fold,
                val_fold,
                epochs=config.gnn_epochs,
                lr=config.gnn_lr,
                device=config.device,
                patience=config.gnn_patience
            )
        
        cv_results.append({
            'model': model,
            'val_f1': best_f1,
            'val_auc': best_auc
        })
        
        print(f"  ✅ 第{fold+1}折: F1={best_f1:.4f}, AUC-PR={best_auc:.4f}")
    
    # 选择最佳模型
    best_result = max(cv_results, key=lambda x: x['val_f1'])
    avg_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_auc = np.mean([r['val_auc'] for r in cv_results])
    
    print(f"\n✅ 交叉验证完成:")
    print(f"  平均F1: {avg_f1:.4f}")
    print(f"  平均AUC: {avg_auc:.4f}")
    print(f"  最佳F1: {best_result['val_f1']:.4f}")
    
    return best_result['model'], avg_f1, avg_auc, cv_results


def train_and_test_ultimate(train_file, test_files, config):
    """ULTIMATE训练-测试流程"""
    train_name = os.path.splitext(os.path.basename(train_file))[0]
    
    print(f"\n{'='*80}")
    print(f"🚀 ULTIMATE PIPELINE: {train_name}")
    print(f"{'='*80}")
    
    # 创建输出目录
    ratio_str = f"{config.target_ratio:.2f}".replace(".", "")
    output_dir_name = f"{train_name}_ultimate_r{ratio_str}"
    output_path = os.path.join(config.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    total_start_time = time.time()
    
    # ============ 阶段1: 加载数据 ============
    print(f"\n📊 阶段1: 加载训练数据")
    train_dataset = load_dataset_quiet(train_file, config)
    
    if not train_dataset:
        print(f"❌ 数据集为空")
        return None
    
    print(f"✅ 加载了 {len(train_dataset)} 个蛋白质")
    orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_dataset)
    print(f"📊 原始数据: {orig_pos:,} 正 / {orig_neg:,} 负 (比例: {orig_ratio:.3%})")
    
    # ============ 阶段2: 训练扩散模型 ============
    diffusion_model = train_enhanced_diffusion_model(train_dataset, config)
    
    # ============ 阶段3: 加载边预测器 ============
    edge_predictor = load_edge_predictor(config)
    if edge_predictor is None:
        print(f"❌ 边预测器加载失败")
        return None
    
    # ============ 阶段4: 数据增强 ============
    print(f"\n🛡️  阶段4: ULTIMATE 数据增强")
    augment_start = time.time()
    
    augmented_data, aug_stats = ultimate_augment_dataset(
        train_dataset,
        diffusion_model,
        edge_predictor,
        config
    )
    
    augment_time = time.time() - augment_start
    aug_ratio, aug_pos, aug_neg = calculate_class_ratio(augmented_data)
    
    print(f"\n✅ 增强完成: {aug_pos:,} 正 / {aug_neg:,} 负 (比例: {aug_ratio:.3%})")
    print(f"   用时: {augment_time:.1f}秒")
    print(f"   质量: {aug_stats['avg_quality']:.3f}")
    print(f"   多样性: {aug_stats['avg_diversity']:.3f}")
    
    # ============ 阶段5: 训练GNN ============
    print(f"\n🔄 阶段5: 交叉验证训练")
    train_start = time.time()
    
    best_model, avg_f1, avg_auc, cv_results = cross_validation_training(
        augmented_data,
        train_dataset,
        config
    )
    
    train_time = time.time() - train_start
    print(f"   用时: {train_time:.1f}秒")
    
    # 保存模型
    model_path = os.path.join(output_path, "ultimate_gnn_model.pt")
    torch.save(best_model.state_dict(), model_path)
    print(f"💾 模型已保存: {model_path}")
    
    # ============ 阶段6: 测试 ============
    print(f"\n🔍 阶段6: 模型测试")
    print(f"{'='*80}")
    
    test_results = {}
    
    for test_file in test_files:
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n📊 测试: {test_name}")
        
        try:
            test_dataset = load_dataset_quiet(test_file, config)
            print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")
            
            # 评估
            start_time = time.time()
            
            if ADVANCED_GNN_AVAILABLE and config.use_advanced_gnn:
                from torch_geometric.loader import DataLoader
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                metrics = best_model.evaluate(test_loader, config.device)
            else:
                metrics = best_model.evaluate(test_dataset, device=config.device)
            
            eval_time = time.time() - start_time
            
            print(f"📈 结果 ({eval_time:.2f}s):")
            print(f"  F1:      {metrics['f1']:.4f}")
            print(f"  MCC:     {metrics['mcc']:.4f}")
            print(f"  ACC:     {metrics['accuracy']:.4f}")
            print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            
            test_results[test_name] = metrics
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ============ 保存结果 ============
    total_time = time.time() - total_start_time
    
    full_results = {
        "pipeline": "ULTIMATE",
        "model_name": train_name,
        "training_info": {
            "original_positive": orig_pos,
            "original_negative": orig_neg,
            "original_ratio": orig_ratio,
            "augmented_positive": aug_pos,
            "augmented_negative": aug_neg,
            "augmented_ratio": aug_ratio,
            "target_ratio": config.target_ratio,
            "cv_avg_f1": avg_f1,
            "cv_avg_auc": avg_auc,
            "augmentation_stats": aug_stats,
            "total_time": total_time
        },
        "test_results": test_results
    }
    
    result_file = os.path.join(output_path, "ultimate_results.json")
    with open(result_file, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"\n✅ ULTIMATE训练-测试完成!")
    print(f"⏱️  总用时: {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"📊 结果已保存: {result_file}")
    
    return full_results


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"🚀 ULTIMATE PIPELINE 启动")
    print(f"{'='*80}")
    
    # 配置
    config = UltimateConfig(target_ratio=0.5, experiment_name="ultimate_v1")
    set_seed(config.seed)
    
    # 查找数据文件
    train_files = sorted(glob.glob(os.path.join(config.data_dir, "*Train*.txt")))
    test_files = sorted(glob.glob(os.path.join(config.data_dir, "*Test*.txt")))
    test_files = [f for f in test_files if 'DNA-' in os.path.basename(f)]
    
    print(f"\n🔍 找到 {len(train_files)} 个训练文件")
    print(f"   找到 {len(test_files)} 个测试文件")
    
    # 运行pipeline
    all_results = {}
    
    for train_file in train_files:
        try:
            result = train_and_test_ultimate(train_file, test_files, config)
            if result:
                all_results[os.path.basename(train_file)] = result
        except Exception as e:
            print(f"\n❌ Pipeline失败 ({train_file}): {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    if all_results:
        summary_file = os.path.join(config.output_dir, "ultimate_pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"🎉 ULTIMATE PIPELINE 完成!")
        print(f"{'='*80}")
        print(f"📊 汇总结果: {summary_file}")


if __name__ == "__main__":
    main()
