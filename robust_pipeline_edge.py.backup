#!/usr/bin/env python3
"""
使用PPI边预测器的鲁棒性增强训练-测试管道

边预测器优势 (超稳定版 v3.0):
  1. 基于真实PPI知识: 在1,858,944条蛋白质相互作用数据上训练 (STRING v12.0)
  2. 高准确度: AUC=0.9300 (训练), 0.9297 (测试)，性能优异！
  3. 混合评估机制: 边预测分数 + 余弦相似度 + 欧氏距离
  4. Top-K保证: 确保每个节点至少有k个邻接边
  5. 更好的泛化能力: 学到的PPI关系可迁移到不同蛋白质
  6. 训练稳定: 66轮无崩溃，相比之前版本提升+3.12%
"""
import os
import time
import glob
import json
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from balanced_training_config import BalancedTrainingConfig
from improved_gnn_model import ImprovedBindingSiteGNN
from data_loader import ProteinDataset
from ddpm_diffusion_model import EnhancedDiffusionModel
from main import calculate_class_ratio
from gnn_model import set_seed
from edge_predictor_augmentation import (
    ImprovedEdgePredictor,
    robust_augment_dataset_with_edge_predictor,
    build_edges_with_edge_predictor
)


class RobustTrainingConfigWithEdgePredictor(BalancedTrainingConfig):
    """使用边预测器的鲁棒训练配置"""
    def __init__(self, target_ratio=0.9, experiment_name="default", use_edge_predictor=True):
        super().__init__()
        self.target_ratio = target_ratio
        self.experiment_name = experiment_name
        self.min_samples_per_protein = 5
        self.max_augment_ratio = 2.0

        # 质量控制
        self.quality_threshold = 0.7
        self.diversity_threshold = 0.3

        # 域适应
        self.use_domain_adaptation = True
        self.domain_weight = 0.1

        # 交叉验证
        self.use_cross_validation = True
        self.cv_folds = 3

        # 集成学习
        self.ensemble_size = 3

        # 边预测器配置（🚀 GPU优化版 - 平衡性能与质量）
        self.use_edge_predictor = use_edge_predictor
        self.edge_predictor_config = {
            'predictor_threshold': 0.8,   # 🚀 较严格（平衡速度和边质量）
            'sim_threshold': 0.7,         # 🚀 适中的相似度要求
            'dist_threshold': 1.2,        # 🚀 适中的距离限制
            'top_k': 5,                   # 保证基本连通性
            'connect_generated_nodes': True,   # ✅ 保留（增强图连通性）
            'use_topk_guarantee': True
        }

        # 输出控制
        self.verbose_loading = False

        print(f"🎯 鲁棒训练配置:")
        print(f"  - 目标比例: {self.target_ratio:.1%}")
        print(f"  - 使用边预测器: {self.use_edge_predictor}")
        print(f"  - 边预测阈值: {self.edge_predictor_config['predictor_threshold']}")
        print(f"  - Top-K保证: {self.edge_predictor_config['top_k']}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")


def load_or_train_edge_predictor(
    train_dataset,
    config,
    pretrained_path=None
):
    """
    加载或训练边预测器

    Args:
        train_dataset: 训练数据集
        config: 配置对象
        pretrained_path: 预训练模型路径

    Returns:
        edge_predictor: 训练好的或加载的边预测器
    """
    print(f"🔗 初始化边预测器...")

    # 获取特征维度
    feature_dim = train_dataset[0].x.size(1)

    # 尝试加载预训练模型
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"✅ 加载预训练边预测器: {pretrained_path}")
        try:
            # 加载checkpoint查看hidden_dim
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            # 从checkpoint推断hidden_dim（假设保存在state_dict中）
            # fc_transform.weight shape: [hidden_dim, input_dim]
            if 'fc_transform.weight' in checkpoint:
                hidden_dim = checkpoint['fc_transform.weight'].shape[0]
            else:
                # 如果是完整的checkpoint格式
                hidden_dim = checkpoint['model_state_dict']['fc_transform.weight'].shape[0] if 'model_state_dict' in checkpoint else 1024

            edge_predictor = ImprovedEdgePredictor(
                input_dim=feature_dim,
                hidden_dim=hidden_dim
            )

            # 加载权重
            if 'model_state_dict' in checkpoint:
                edge_predictor.load_state_dict(checkpoint['model_state_dict'])
            else:
                edge_predictor.load_state_dict(checkpoint)

            edge_predictor.eval()
            print(f"   模型配置: input_dim={feature_dim}, hidden_dim={hidden_dim}")
            return edge_predictor
        except Exception as e:
            print(f"⚠️  加载预训练模型失败: {e}")
            print(f"   使用随机初始化")

    # 如果没有预训练模型，使用默认配置
    edge_predictor = ImprovedEdgePredictor(
        input_dim=feature_dim,
        hidden_dim=358  # 默认值
    )

    # 如果没有预训练模型，使用随机初始化
    # (实际场景中应该在有标注边的数据上训练)
    print(f"⚠️  未找到预训练模型，使用随机初始化")
    print(f"   建议: 在蛋白质图数据上预先训练边预测器")
    edge_predictor.to(config.device)
    edge_predictor.eval()

    return edge_predictor


def domain_adaptive_loss(predictions, targets, domain_weight=0.1):
    """域适应损失"""
    if predictions.dim() == 1:
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets.float())
    else:
        targets = targets.long()
        base_loss = torch.nn.functional.cross_entropy(predictions, targets)

    batch_size = predictions.size(0)
    if batch_size > 1:
        if predictions.dim() == 1:
            probs = torch.sigmoid(predictions)
            prob_var = torch.var(probs, dim=0)
        else:
            probs = torch.softmax(predictions, dim=1)
            prob_var = torch.var(probs, dim=0).mean()
        domain_loss = domain_weight * prob_var
    else:
        domain_loss = 0.0

    return base_loss + domain_loss


class RobustGNNModel(ImprovedBindingSiteGNN):
    """鲁棒GNN模型"""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3, use_focal_loss=True,
                 focal_alpha=0.75, focal_gamma=2.0, pos_weight=3.0, domain_weight=0.1):
        super().__init__(input_dim, hidden_dim, dropout, use_focal_loss,
                         focal_alpha, focal_gamma, pos_weight)
        self.domain_weight = domain_weight

    def train_with_domain_adaptation(self, train_data, val_data, epochs=100, lr=0.001,
                                    device='cuda', patience=10):
        """域适应训练"""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_val_f1 = 0
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for data in train_data:
                data = data.to(device)
                optimizer.zero_grad()

                out = self(data)
                loss = domain_adaptive_loss(out, data.y, self.domain_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                val_metrics = self.evaluate(val_data, device)
                scheduler.step(val_metrics['f1'])

                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    best_val_auc = val_metrics['auc_pr']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        return best_val_auc, best_val_f1


def cross_validation_training_with_edge_predictor(
    augmented_data,
    original_data,
    config,
    edge_predictor=None
):
    """交叉验证训练（使用边预测器构建的图）"""
    print(f"\n🔄 {config.cv_folds}折交叉验证训练...")
    print(f"📊 训练策略: 使用边预测器构建的增强图")

    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_results = []
    original_indices = list(range(len(original_data)))

    for fold, (_, val_idx) in enumerate(kf.split(original_indices)):
        print(f"\n📊 第 {fold+1}/{config.cv_folds} 折")

        val_original_indices = set(val_idx)
        train_original = [original_data[i] for i in range(len(original_data))
                         if i not in val_original_indices]
        train_fold = augmented_data + train_original

        val_fold = [original_data[i] for i in val_idx]

        print(f"  📈 训练集大小: {len(train_fold)} (增强: {len(augmented_data)}, "
              f"原始: {len(train_original)})")
        print(f"  📊 验证集大小: {len(val_fold)} (仅原始数据)")

        model = RobustGNNModel(
            input_dim=config.diffusion_input_dim,
            hidden_dim=config.gnn_hidden_dim,
            dropout=config.gnn_dropout,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            pos_weight=config.pos_weight,
            domain_weight=config.domain_weight
        )

        best_auc, best_f1 = model.train_with_domain_adaptation(
            train_fold, val_fold,
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

    return cv_results


def train_and_test_with_edge_predictor(
    train_file,
    test_files,
    config,
    edge_predictor=None,
    new_test_files=None
):
    """使用边预测器的训练-测试流程"""
    train_name = os.path.splitext(os.path.basename(train_file))[0]
    print(f"\n🚀 开始鲁棒训练-测试 (使用边预测器): {train_name}")
    print("="*60)

    ratio_str = f"{config.target_ratio:.2f}".replace(".", "")
    output_dir_name = f"{train_name}_robust_r{ratio_str}_edgepred_{config.experiment_name}"
    output_path = os.path.join(config.output_dir, output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 加载数据
    print(f"📊 阶段1: 加载训练数据...")
    train_dataset = load_dataset_quiet(train_file, config)

    if not train_dataset:
        print(f"❌ 数据集为空: {train_file}")
        return None

    print(f"✅ 加载了 {len(train_dataset)} 个蛋白质")

    orig_ratio, orig_pos, orig_neg = calculate_class_ratio(train_dataset)
    print(f"📊 原始数据: {orig_pos:,} 正样本, {orig_neg:,} 负样本 (比例: {orig_ratio:.3%})")

    # 训练扩散模型
    print(f"\n🧠 阶段2: 训练扩散模型...")
    diffusion_model = EnhancedDiffusionModel(
        input_dim=config.diffusion_input_dim,
        T=config.diffusion_T,
        device=config.device
    )

    diffusion_start = time.time()
    diffusion_model.train_on_positive_samples(
        train_dataset,
        epochs=config.diffusion_epochs,
        batch_size=config.diffusion_batch_size
    )
    diffusion_time = time.time() - diffusion_start
    print(f"✅ 扩散模型训练完成: {diffusion_time:.1f}秒")

    # 加载或初始化边预测器
    print(f"\n🔗 阶段2.5: 初始化边预测器 (超稳定版)...")
    if edge_predictor is None:
        # 使用训练好的PPI边预测器模型 - 超稳定版 (AUC=0.9300)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ppi_model_path = os.path.join(current_dir, "models", "edge_predictor_best_ultra_stable.pth")

        print(f"📁 使用超稳定版PPI模型 (训练AUC: 0.9300, 测试AUC: 0.9297)")

        edge_predictor = load_or_train_edge_predictor(
            train_dataset,
            config,
            pretrained_path=ppi_model_path
        )

        # 🚀 GPU优化：确保边预测器在GPU上
        edge_predictor = edge_predictor.to(config.device)
        edge_predictor.eval()
        print(f"✅ 边预测器已加载到GPU: {config.device}")

    # 使用边预测器进行鲁棒增强
    print(f"\n🛡️ 阶段3: 使用边预测器的鲁棒增强...")
    augment_start = time.time()
    augmented_data, aug_stats = robust_augment_dataset_with_edge_predictor(
        train_dataset,
        diffusion_model,
        edge_predictor,
        config,
        predictor_config=config.edge_predictor_config
    )
    augment_time = time.time() - augment_start

    aug_ratio, aug_pos, aug_neg = calculate_class_ratio(augmented_data)
    print(f"✅ 增强完成: {aug_pos:,} 正样本, {aug_neg:,} 负样本 "
          f"(比例: {aug_ratio:.3%}) - 用时: {augment_time:.1f}秒")

    # 交叉验证训练
    print(f"\n🔄 阶段4: 交叉验证训练...")
    gnn_start = time.time()
    cv_results = cross_validation_training_with_edge_predictor(
        augmented_data,
        train_dataset,
        config,
        edge_predictor
    )
    gnn_time = time.time() - gnn_start

    best_cv_result = max(cv_results, key=lambda x: x['val_f1'])
    best_model = best_cv_result['model']

    avg_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_auc = np.mean([r['val_auc'] for r in cv_results])

    print(f"✅ 交叉验证完成: 平均F1={avg_f1:.4f}, 平均AUC-PR={avg_auc:.4f} - "
          f"用时: {gnn_time:.1f}秒")

    # 保存模型
    model_save_path = os.path.join(output_path, "robust_gnn_model.pt")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"💾 最佳模型保存至: {model_save_path}")

    # 测试
    print(f"\n🔍 阶段5: 模型测试")
    print("="*60)

    test_results = {}

    print(f"\n{'='*80}")
    print(f"📊 原始测试集 (DNA系列)")
    print(f"{'='*80}")

    for test_file in test_files:
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f"\n📊 测试数据集: {test_name}")

        try:
            test_dataset = load_dataset_quiet(test_file, config)
            print(f"✅ 加载了 {len(test_dataset)} 个蛋白质")

            start_time = time.time()
            metrics = best_model.evaluate(test_dataset, device=config.device)
            eval_time = time.time() - start_time

            print(f"📈 鲁棒测试结果 ({eval_time:.2f}s):")
            print(f"  🎯 F1 Score:         {metrics['f1']:.4f}")
            print(f"  🎯 MCC:              {metrics['mcc']:.4f}")
            print(f"  🎯 Accuracy:         {metrics['accuracy']:.4f}")
            print(f"  🎯 AUC-PR:           {metrics['auc_pr']:.4f}")
            print(f"  🎯 AUC-ROC:          {metrics['auc_roc']:.4f}")

            test_results[test_name] = metrics

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")

    # 保存结果
    total_time = time.time() - diffusion_start

    full_results = {
        "model_name": train_name + "_robust_edgepred",
        "model_type": "Robust GNN with Edge Predictor",
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
            "diffusion_time": diffusion_time,
            "augment_time": augment_time,
            "gnn_time": gnn_time,
            "total_time": total_time
        },
        "edge_predictor_config": config.edge_predictor_config,
        "test_results": test_results
    }

    with open(os.path.join(output_path, "robust_results_edgepred.json"), 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n✅ 训练-测试完成!")
    print(f"⏱️ 总用时: {total_time//60:.0f}m {total_time%60:.0f}s")

    return full_results


def load_dataset_quiet(dataset_file, config):
    """静默加载数据集"""
    import shutil
    import contextlib

    temp_data_dir = os.path.join(config.data_dir, "temp")
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


def main():
    """主函数"""
    print("🛡️ 使用边预测器的鲁棒管道启动")
    print("="*80)
    print("改进策略: 边预测 + 质量控制 + 域适应 + 交叉验证")
    print("="*80)

    config = RobustTrainingConfigWithEdgePredictor(use_edge_predictor=True)
    set_seed(config.seed)

    print(f"\n🛡️ 配置:")
    print(f"  - 使用边预测器: {config.use_edge_predictor}")
    print(f"  - 边预测阈值: {config.edge_predictor_config['predictor_threshold']}")

    train_files = sorted(glob.glob(os.path.join(config.data_dir, "*Train*.txt")))
    test_files = sorted(glob.glob(os.path.join(config.data_dir, "*Test*.txt")))

    original_test_files = [f for f in test_files if 'DNA-' in os.path.basename(f)]
    new_test_files = sorted(glob.glob(os.path.join(config.data_dir, "PDNA-*-test.txt")))

    print(f"\n🔍 找到 {len(train_files)} 个训练文件")
    print(f"   - 原始测试集 (DNA系列): {len(original_test_files)} 个")

    all_results = {}
    total_start = time.time()

    for train_file in train_files:
        try:
            result = train_and_test_with_edge_predictor(
                train_file,
                original_test_files,
                config
            )
            if result:
                all_results[os.path.splitext(os.path.basename(train_file))[0]] = result
        except Exception as e:
            print(f"❌ 处理失败 {train_file}: {str(e)}")

    total_pipeline_time = time.time() - total_start

    if all_results:
        results_file = os.path.join(config.output_dir, "robust_pipeline_edgepred_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n🎉 鲁棒管道执行完成!")
        print(f"⏱️ 总时间: {total_pipeline_time//3600:.0f}h {(total_pipeline_time%3600)//60:.0f}m")
        print(f"📊 详细结果: {results_file}")


if __name__ == "__main__":
    main()
