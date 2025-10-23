#!/usr/bin/env python3
"""
Ultimate 数据增强模块

整合:
1. 增强版条件扩散模型
2. 边预测器图构建
3. 自适应质量控制
4. 多样性保证
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# 导入我们创建的增强版模型
try:
    from enhanced_diffusion_model import EnhancedConditionalDiffusionModel
    ENHANCED_AVAILABLE = True
except ImportError:
    print("⚠️  增强版扩散模型不可用，使用标准版本")
    from ddpm_diffusion_model import EnhancedDiffusionModel
    ENHANCED_AVAILABLE = False

from edge_predictor_augmentation import ImprovedEdgePredictor, build_edges_with_edge_predictor


def ultimate_augment_protein(
    protein_data,
    diffusion_model,
    edge_predictor,
    config,
    target_ratio=0.9
):
    """
    终极数据增强 - 单个蛋白质
    
    Args:
        protein_data: PyG Data object
        diffusion_model: 训练好的扩散模型
        edge_predictor: 训练好的边预测器
        config: UltimateConfig对象
        target_ratio: 目标正样本比例
        
    Returns:
        augmented_data: 增强后的Data对象
        stats: 统计信息
    """
    # 计算需要生成的数量
    n_pos = (protein_data.y == 1).sum().item()
    n_neg = (protein_data.y == 0).sum().item()
    total_nodes = n_pos + n_neg
    
    target_pos = int(total_nodes * target_ratio)
    n_to_generate_base = max(config.min_samples_per_protein, target_pos - n_pos)
    
    # 🚀 关键优化：生成多倍候选样本
    sample_multiplier = config.enhanced_diffusion_config.get('sample_multiplier', 10)
    n_to_generate = n_to_generate_base * sample_multiplier
    
    if n_to_generate <= 0:
        return protein_data, {'num_generated': 0, 'quality': 0, 'diversity': 0}
    
    # 🚀 GPU加速：所有操作保持在GPU上
    device = config.device

    # 生成样本
    if ENHANCED_AVAILABLE and config.use_enhanced_diffusion:
        # 使用增强版条件扩散（返回GPU张量）
        generated_samples, quality_scores = diffusion_model.generate_positive_sample(
            protein_data,
            num_samples=n_to_generate,
            quality_threshold=config.enhanced_diffusion_config['quality_threshold'],
            max_attempts=config.enhanced_diffusion_config['max_attempts']
        )
        # 确保在正确的设备上
        generated_samples = generated_samples.to(device)
        quality_scores = quality_scores.to(device)
    else:
        # 使用标准扩散模型
        generated_samples = diffusion_model.generate_positive_sample(
            protein_data.protein_context,
            num_samples=n_to_generate
        )
        if generated_samples is None or len(generated_samples) == 0:
            return protein_data, {'num_generated': 0, 'quality': 0, 'diversity': 0}

        # 评估质量（GPU版本）
        generated_samples_tensor = torch.tensor(generated_samples, device=device, dtype=torch.float32)
        positive_samples = protein_data.x[protein_data.y == 1].to(device) if n_pos > 0 else protein_data.x.to(device)
        quality_scores = evaluate_sample_quality_gpu(generated_samples_tensor, positive_samples, config.quality_threshold)
        generated_samples = generated_samples_tensor

    # 🚀 质量过滤（GPU操作）
    quality_mask = quality_scores > config.quality_threshold
    if quality_mask.sum() == 0:
        # 如果没有高质量样本，降低阈值
        quality_mask = quality_scores > (config.quality_threshold * 0.7)

    filtered_samples = generated_samples[quality_mask]
    filtered_scores = quality_scores[quality_mask]

    # 🚀 如果过滤后样本太少，选择top-k（GPU操作）
    if filtered_samples.size(0) < n_to_generate_base:
        top_k_indices = torch.topk(quality_scores, k=min(n_to_generate_base, quality_scores.size(0))).indices
        filtered_samples = generated_samples[top_k_indices]
        filtered_scores = quality_scores[top_k_indices]
    elif filtered_samples.size(0) > n_to_generate_base * 2:
        # 如果样本太多，选择最优的（GPU操作）
        top_k_indices = torch.topk(filtered_scores, k=n_to_generate_base).indices
        filtered_samples = filtered_samples[top_k_indices]
        filtered_scores = filtered_scores[top_k_indices]

    # 已经是torch tensor，保持在GPU上
    generated_x = filtered_samples
    generated_y = torch.ones(generated_x.size(0), dtype=torch.long, device=device)
    
    # 使用边预测器构建图
    augmented_graph, edge_stats = build_edges_with_edge_predictor(
        protein_data,
        generated_x,
        edge_predictor,
        config.device,
        **config.edge_predictor_config
    )
    
    # 🚀 计算多样性（GPU操作）
    if filtered_samples.size(0) > 1:
        diversity = calculate_diversity_gpu(filtered_samples)
    else:
        diversity = 0.5

    stats = {
        'num_generated': filtered_samples.size(0),
        'num_candidates': generated_samples.size(0),
        'quality': float(filtered_scores.mean().item()),  # GPU → scalar
        'diversity': diversity,
        'num_edges': edge_stats.get('total_edges', 0),
        'avg_edge_score': edge_stats.get('avg_edge_score', 0)
    }
    
    return augmented_graph, stats


def ultimate_augment_dataset(
    dataset,
    diffusion_model,
    edge_predictor,
    config
):
    """
    终极数据增强 - 整个数据集
    
    Returns:
        augmented_dataset: 增强后的数据集
        global_stats: 全局统计信息
    """
    print(f"\n🚀 ULTIMATE 数据增强开始")
    print(f"  策略: 增强版扩散 + 边预测器 + 质量控制")
    print(f"  目标比例: {config.target_ratio:.1%}")
    print(f"  质量阈值: {config.quality_threshold}")
    print(f"  采样倍数: {config.enhanced_diffusion_config.get('sample_multiplier', 10)}x")
    
    augmented_dataset = []
    all_stats = []
    
    for protein_data in tqdm(dataset, desc="Ultimate augmenting"):
        try:
            aug_data, stats = ultimate_augment_protein(
                protein_data,
                diffusion_model,
                edge_predictor,
                config,
                target_ratio=config.target_ratio
            )
            
            augmented_dataset.append(aug_data)
            all_stats.append(stats)
            
        except Exception as e:
            print(f"⚠️  增强失败 ({protein_data.name if hasattr(protein_data, 'name') else 'unknown'}): {e}")
            augmented_dataset.append(protein_data)
            all_stats.append({'num_generated': 0, 'quality': 0, 'diversity': 0})
    
    # 全局统计
    total_generated = sum(s['num_generated'] for s in all_stats)
    avg_quality = np.mean([s['quality'] for s in all_stats if s['quality'] > 0])
    avg_diversity = np.mean([s['diversity'] for s in all_stats if s['diversity'] > 0])
    total_edges = sum(s.get('num_edges', 0) for s in all_stats)
    
    global_stats = {
        'total_proteins': len(dataset),
        'total_generated': total_generated,
        'avg_quality': avg_quality,
        'avg_diversity': avg_diversity,
        'total_edges': total_edges,
        'success_rate': sum(1 for s in all_stats if s['num_generated'] > 0) / len(all_stats)
    }
    
    print(f"\n✅ ULTIMATE 增强完成:")
    print(f"  总蛋白质数: {global_stats['total_proteins']}")
    print(f"  生成样本数: {global_stats['total_generated']:,}")
    print(f"  平均质量: {global_stats['avg_quality']:.3f}")
    print(f"  平均多样性: {global_stats['avg_diversity']:.3f}")
    print(f"  成功率: {global_stats['success_rate']:.1%}")
    print(f"  总边数: {global_stats['total_edges']:,}")
    
    return augmented_dataset, global_stats


def evaluate_sample_quality(generated_samples, positive_samples, threshold=0.5):
    """评估生成样本质量（用于标准扩散模型，NumPy版本）"""
    if len(positive_samples) == 0:
        return np.ones(len(generated_samples)) * 0.5

    pos_mean = np.mean(positive_samples, axis=0)
    pos_std = np.std(positive_samples, axis=0) + 1e-6

    # 归一化距离
    normalized = (generated_samples - pos_mean) / pos_std
    dist = np.mean(normalized ** 2, axis=1)

    # 转换为质量分数
    quality = 1.0 / (1.0 + dist)

    return quality


def evaluate_sample_quality_gpu(generated_samples, positive_samples, threshold=0.5):
    """🚀 评估生成样本质量（GPU加速版本）"""
    if positive_samples.size(0) == 0:
        return torch.ones(generated_samples.size(0), device=generated_samples.device) * 0.5

    pos_mean = torch.mean(positive_samples, dim=0)
    pos_std = torch.std(positive_samples, dim=0) + 1e-6

    # 归一化距离（GPU操作）
    normalized = (generated_samples - pos_mean) / pos_std
    dist = torch.mean(normalized ** 2, dim=1)

    # 转换为质量分数
    quality = 1.0 / (1.0 + dist)

    return quality


def calculate_diversity(samples):
    """计算样本多样性（NumPy版本）"""
    if len(samples) < 2:
        return 0.5

    # 计算成对余弦相似度
    samples_norm = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(samples_norm, samples_norm.T)

    # 去除对角线（自身相似度）
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)

    # 多样性 = 1 - 平均相似度
    avg_similarity = np.mean(similarity_matrix[mask])
    diversity = 1.0 - avg_similarity

    return max(0.0, min(1.0, diversity))


def calculate_diversity_gpu(samples):
    """🚀 计算样本多样性（GPU加速版本）"""
    if samples.size(0) < 2:
        return 0.5

    # 计算成对余弦相似度（GPU操作）
    samples_norm = F.normalize(samples, p=2, dim=1)  # L2归一化
    similarity_matrix = torch.mm(samples_norm, samples_norm.t())  # 余弦相似度矩阵

    # 去除对角线（自身相似度）
    mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    mask.fill_diagonal_(False)

    # 多样性 = 1 - 平均相似度
    avg_similarity = similarity_matrix[mask].mean().item()
    diversity = 1.0 - avg_similarity

    return max(0.0, min(1.0, diversity))
