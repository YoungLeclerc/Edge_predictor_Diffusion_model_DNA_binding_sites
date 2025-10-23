#!/usr/bin/env python3
"""
Ultimate Pipeline 配置文件

整合所有优化：
- 增强版条件扩散模型
- 高级GAT-GNN模型
- 边预测器
- 自适应数据增强
"""
import torch
from balanced_training_config import BalancedTrainingConfig


class UltimateConfig(BalancedTrainingConfig):
    """终极优化配置"""

    def __init__(self, target_ratio=0.5, experiment_name="ultimate", use_enhanced_diffusion=True):
        super().__init__()

        self.target_ratio = target_ratio
        self.experiment_name = experiment_name

        # ============ 增强版扩散模型配置 ============
        # 🔧 已修复梯度计算bug，重新启用增强版扩散模型
        self.use_enhanced_diffusion = True  # 使用增强版条件扩散模型
        self.enhanced_diffusion_config = {
            'T': 300,  # 扩散步数（平衡质量和速度：200步保证质量，比500步快60%）
            'hidden_dim': 512,
            'context_dim': 256,
            'quality_threshold': 0.5,  # 🔧 保持高质量标准（修复评估函数后分数会提升）
            'max_attempts': 3,  # 最大采样尝试次数（适中，保证质量）
            'sample_multiplier': 5,  # 生成候选样本倍数（保证有足够的候选样本）
        }

        # ============ 高级GNN配置 ============
        self.use_advanced_gnn = True
        self.advanced_gnn_config = {
            'hidden_dim': 256,  # GNN隐藏层维度
            'num_layers': 4,  # GAT层数
            'heads': 4,  # 注意力头数
            'dropout': 0.3,
            'use_edge_features': True,  # 使用边特征
            'focal_alpha': 0.25,  # Focal loss alpha
            'focal_gamma': 2.0,  # Focal loss gamma
            'class_balanced': True,  # 类别平衡损失
        }

        # ============ 边预测器配置 ============
        self.use_edge_predictor = True
        self.edge_predictor_config = {
            'predictor_threshold': 0.8,
            'sim_threshold': 0.7,
            'dist_threshold': 1.2,
            'top_k': 5,
            'connect_generated_nodes': True,
            'use_topk_guarantee': True
        }

        # ============ 训练配置优化 ============
        # 扩散模型训练
        self.diffusion_epochs = 200  # 增强版需要更多轮
        self.diffusion_batch_size = 64
        self.diffusion_lr = 1e-4

        # GNN训练
        self.gnn_epochs = 200  # 从100增加到200
        self.gnn_patience = 20  # 从10增加到20
        self.gnn_lr = 1e-3
        self.gnn_hidden_dim = self.advanced_gnn_config['hidden_dim']
        self.gnn_dropout = self.advanced_gnn_config['dropout']

        # ============ 数据增强配置 ============
        self.min_samples_per_protein = 10  # 每个蛋白质最少生成样本数
        self.max_augment_ratio = 5.0  # 最大增强倍数

        # 质量控制（放宽以获得更多样本）
        self.quality_threshold = 0.8  # 
        self.diversity_threshold = 0.6

        # ============ 交叉验证 ============
        self.use_cross_validation = True
        self.cv_folds = 3

        # ============ 集成学习 ============
        self.use_ensemble = True
        self.ensemble_size = 3

        # ============ 输出控制 ============
        self.verbose_loading = False
        self.save_intermediate = True  # 保存中间结果

        # ============ GPU优化 ============
        # 确保所有模型在GPU上
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        print("="*80)
        print("🚀 ULTIMATE PIPELINE 配置")
        print("="*80)
        print(f"📊 数据增强:")
        print(f"  - 目标比例: {self.target_ratio:.1%}")
        print(f"  - 质量阈值: {self.quality_threshold}")
        print(f"  - 采样倍数: {self.enhanced_diffusion_config['sample_multiplier']}x")
        print(f"")
        print(f"🧠 增强版扩散模型:")
        print(f"  - 启用: {self.use_enhanced_diffusion}")
        print(f"  - 扩散步数 T: {self.enhanced_diffusion_config['T']}")
        print(f"  - 上下文维度: {self.enhanced_diffusion_config['context_dim']}")
        print(f"  - 最大尝试: {self.enhanced_diffusion_config['max_attempts']}")
        print(f"")
        print(f"🔗 高级GAT-GNN:")
        print(f"  - 启用: {self.use_advanced_gnn}")
        print(f"  - 隐藏层: {self.advanced_gnn_config['hidden_dim']}")
        print(f"  - GAT层数: {self.advanced_gnn_config['num_layers']}")
        print(f"  - 注意力头: {self.advanced_gnn_config['heads']}")
        print(f"  - 训练轮数: {self.gnn_epochs}")
        print(f"")
        print(f"🔗 边预测器:")
        print(f"  - 启用: {self.use_edge_predictor}")
        print(f"  - 阈值: {self.edge_predictor_config['predictor_threshold']}")
        print(f"  - Top-K: {self.edge_predictor_config['top_k']}")
        print(f"")
        print(f"🎯 训练策略:")
        print(f"  - 交叉验证: {self.cv_folds}折")
        print(f"  - 集成学习: {self.ensemble_size}个模型")
        print(f"  - 设备: {self.device}")
        print(f"")
        print(f"📁 输出目录: {self.output_dir}")
        print("="*80)
