#!/usr/bin/env python3
"""
使用边预测器构建图的改进数据增强模块
特点:
  1. 使用训练好的边预测器替代KNN构建边
  2. 混合策略: 边预测分数 + 余弦相似度 + 距离阈值
  3. 保证最少连接数 (Top-K)
  4. 支持增强节点间的连接和与原始节点的连接
  5. 🔧 完整的边界检查和错误处理
"""
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


class ImprovedEdgePredictor(nn.Module):
    """改进的边预测器 - 支持权重加载和多种连接模式"""
    def __init__(self, input_dim, hidden_dim=358):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_transform = nn.Linear(input_dim, hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, xi, xj):
        """
        Args:
            xi: 源节点特征 (batch_size_i, input_dim) 或 (batch_size_i, batch_size_j, input_dim)
            xj: 目标节点特征 (batch_size_j, input_dim)

        Returns:
            边存在概率 (batch_size_i * batch_size_j, 1)
        """
        xi_transformed = self.fc_transform(xi)
        xj_transformed = self.fc_transform(xj)

        # 处理不同大小的批量
        if xi_transformed.dim() == 2 and xj_transformed.dim() == 2:
            if xi_transformed.size(0) != xj_transformed.size(0):
                # 扩展为所有组合
                xi_ext = xi_transformed.unsqueeze(1).repeat(1, xj_transformed.size(0), 1)
                xi_ext = xi_ext.view(-1, xi_ext.size(-1))
                xj_ext = xj_transformed.unsqueeze(0).repeat(xi_transformed.size(0), 1, 1)
                xj_ext = xj_ext.view(-1, xj_ext.size(-1))
                x_pair = torch.cat([xi_ext, xj_ext], dim=-1)
            else:
                # 逐行对应
                x_pair = torch.cat([xi_transformed, xj_transformed], dim=-1)
        else:
            x_pair = torch.cat([xi_transformed, xj_transformed], dim=-1)

        return self.model(x_pair)


def build_edges_with_edge_predictor(
    original_data,
    generated_x,
    edge_predictor,
    device,
    predictor_threshold=0.5,
    sim_threshold=0.6,
    dist_threshold=1.5,
    top_k=5,
    connect_generated_nodes=True,
    use_topk_guarantee=True,
    verbose=False
):
    """
    使用边预测器为增强后的图构建边

    Args:
        original_data: 原始图数据 (PyG Data object)
        generated_x: 生成的新节点特征 (num_new_nodes, feature_dim)
        edge_predictor: 训练好的边预测器模型
        device: 计算设备
        predictor_threshold: 边预测器概率阈值
        sim_threshold: 余弦相似度阈值
        dist_threshold: 欧氏距离阈值
        top_k: 保证至少连接的邻接数
        connect_generated_nodes: 是否连接生成节点间的边
        use_topk_guarantee: 是否使用Top-K保证
        verbose: 是否打印调试信息

    Returns:
        augmented_data: 包含新节点和新边的图数据
    """
    # 🔧 修复：处理空数据的情况
    if original_data.x is None or generated_x is None:
        return original_data, {'num_new_edges': 0, 'total_edges': 0, 'edge_scores': []}

    if generated_x.size(0) == 0:
        return original_data, {'num_new_edges': 0, 'total_edges': original_data.edge_index.size(1), 'edge_scores': []}

    original_x = original_data.x.to(device)
    generated_x = generated_x.to(device)

    # 🔧 修复：处理空边列表的情况
    if original_data.edge_index.numel() == 0:
        original_edges = []
    else:
        original_edges = original_data.edge_index.t().tolist()

    new_node_start_idx = original_x.size(0)
    num_new_nodes = generated_x.size(0)

    edge_predictor.eval()
    new_edges = []
    edge_scores = []  # 记录边的预测分数

    if verbose:
        print(f"🔗 开始构建新边...")
        print(f"   原始节点数: {original_x.size(0)}")
        print(f"   生成节点数: {num_new_nodes}")

    # ==========================================
    # 第1部分: 生成节点与原始节点之间的连接
    # ==========================================
    with torch.no_grad():
        for i_new in range(num_new_nodes):
            x_new = generated_x[i_new].unsqueeze(0)  # (1, feature_dim)

            # 扩展x_new来与所有原始节点计算
            x_new_expanded = x_new.repeat(original_x.size(0), 1)  # (num_orig, feature_dim)

            # 1. 边预测器分数
            pred_scores = edge_predictor(x_new_expanded, original_x).squeeze()  # (num_orig,)
            if pred_scores.dim() == 0:
                pred_scores = pred_scores.unsqueeze(0)

            # 2. 余弦相似度
            cos_sim = cosine_similarity(x_new_expanded, original_x).squeeze()  # (num_orig,)
            if cos_sim.dim() == 0:
                cos_sim = cos_sim.unsqueeze(0)

            # 3. 欧氏距离
            dist = torch.norm(x_new_expanded - original_x, dim=1)  # (num_orig,)

            # 选择连接方式
            if use_topk_guarantee:
                # 混合策略: 满足条件 OR Top-K
                condition_mask = (pred_scores > predictor_threshold) & \
                                (cos_sim > sim_threshold) & \
                                (dist < dist_threshold)
                selected_idx = torch.nonzero(condition_mask).squeeze().tolist()
                if isinstance(selected_idx, int):
                    selected_idx = [selected_idx]
                elif not isinstance(selected_idx, list):
                    selected_idx = selected_idx.tolist()

                # 保证Top-K
                topk_vals, topk_idx = torch.topk(
                    pred_scores,
                    k=min(top_k, pred_scores.size(0))
                )
                topk_idx = topk_idx.tolist()

                # 合并: 条件满足的节点 + Top-K节点
                final_idx = set(selected_idx) | set(topk_idx)
            else:
                # 严格条件
                condition_mask = (pred_scores > predictor_threshold) & \
                                (cos_sim > sim_threshold) & \
                                (dist < dist_threshold)
                final_idx = torch.nonzero(condition_mask).squeeze().tolist()
                if isinstance(final_idx, int):
                    final_idx = [final_idx]
                elif not isinstance(final_idx, list):
                    final_idx = final_idx.tolist()

            # 添加双向边
            new_idx = new_node_start_idx + i_new
            for idx in final_idx:
                if 0 <= idx < original_x.size(0):
                    idx = int(idx)
                    # 记录分数
                    try:
                        score = float(pred_scores[idx].cpu().numpy())
                        edge_scores.append(score)
                    except:
                        pass

                    # 添加双向边
                    new_edges.append([idx, new_idx])
                    new_edges.append([new_idx, idx])

            if verbose and i_new % max(1, num_new_nodes // 10) == 0:
                print(f"   已处理 {i_new+1}/{num_new_nodes} 个生成节点, "
                      f"当前连接数: {len(final_idx)}")

    # ==========================================
    # 第2部分: 生成节点之间的连接 (可选)
    # ==========================================
    if connect_generated_nodes and num_new_nodes > 1:
        if verbose:
            print(f"🔗 构建生成节点间的边...")

        with torch.no_grad():
            for i in range(num_new_nodes):
                for j in range(i + 1, num_new_nodes):
                    x_i = generated_x[i].unsqueeze(0)  # (1, feature_dim)
                    x_j = generated_x[j].unsqueeze(0)  # (1, feature_dim)

                    # 计算连接指标
                    pred_score = edge_predictor(x_i, x_j).squeeze()  # scalar
                    cos_sim_val = cosine_similarity(x_i, x_j).squeeze()  # scalar
                    dist_val = torch.norm(x_i - x_j)  # scalar

                    # 判断是否连接
                    if (pred_score > predictor_threshold and
                        cos_sim_val > sim_threshold and
                        dist_val < dist_threshold):

                        i_idx = new_node_start_idx + i
                        j_idx = new_node_start_idx + j

                        new_edges.append([i_idx, j_idx])
                        new_edges.append([j_idx, i_idx])
                        try:
                            edge_scores.append(float(pred_score.cpu().numpy()))
                        except:
                            pass

    # ==========================================
    # 第3部分: 组合所有边和节点
    # ==========================================
    all_x = torch.cat([original_x.cpu(), generated_x.cpu()], dim=0)
    all_y = torch.cat([
        original_data.y.cpu(),
        torch.ones(num_new_nodes, dtype=torch.long)
    ], dim=0)

    # 🔧 修复：验证和清理所有边的索引
    all_edges = original_edges + new_edges
    valid_edges = []
    total_nodes = all_x.size(0)

    for edge in all_edges:
        if len(edge) >= 2:
            src, dst = int(edge[0]), int(edge[1])
            # 检查边是否超出范围
            if 0 <= src < total_nodes and 0 <= dst < total_nodes:
                valid_edges.append([src, dst])
            else:
                # 丢弃无效边
                if verbose:
                    print(f"⚠️  删除无效边: [{src}, {dst}] (总节点数: {total_nodes})")

    # 🔧 修复：如果没有有效边，创建至少一条自循环边避免空图
    if len(valid_edges) == 0:
        valid_edges = [[0, 0]]

    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

    # 创建增强后的图
    augmented_data = Data(
        x=all_x,
        edge_index=edge_index,
        y=all_y
    )

    if verbose:
        print(f"✅ 边构建完成!")
        print(f"   新增边数: {len(new_edges)}")
        print(f"   有效边数: {edge_index.size(1)}")
        if edge_scores:
            print(f"   边预测分数范围: [{np.min(edge_scores):.3f}, {np.max(edge_scores):.3f}]")
            print(f"   边预测分数平均值: {np.mean(edge_scores):.3f}")

    return augmented_data, {
        'num_new_edges': len(new_edges),
        'total_edges': edge_index.size(1),
        'edge_scores': edge_scores
    }


def robust_augment_dataset_with_edge_predictor(
    dataset,
    diffusion_model,
    edge_predictor,
    config,
    predictor_config=None
):
    """
    使用边预测器的鲁棒数据增强

    Args:
        dataset: 原始数据集
        diffusion_model: 扩散模型
        edge_predictor: 边预测器
        config: 训练配置
        predictor_config: 边预测器配置 (dict)
            {
                'predictor_threshold': 0.5,
                'sim_threshold': 0.6,
                'dist_threshold': 1.5,
                'top_k': 5,
                'connect_generated_nodes': True,
                'use_topk_guarantee': True
            }

    Returns:
        augmented_data: 增强后的数据集
        stats: 统计信息
    """
    if predictor_config is None:
        predictor_config = {
            'predictor_threshold': 0.5,
            'sim_threshold': 0.6,
            'dist_threshold': 1.5,
            'top_k': 5,
            'connect_generated_nodes': True,
            'use_topk_guarantee': True
        }

    augmented_data = []
    quality_stats = []
    diversity_stats = []
    edge_stats = []

    print(f"🎯 使用边预测器的鲁棒增强策略:")
    print(f"  - 目标比例: {config.target_ratio:.1%}")
    print(f"  - 质量阈值: {config.quality_threshold}")
    print(f"  - 多样性阈值: {config.diversity_threshold}")
    print(f"  - 边预测阈值: {predictor_config['predictor_threshold']}")
    print(f"  - 相似度阈值: {predictor_config['sim_threshold']}")
    print(f"  - 距离阈值: {predictor_config['dist_threshold']}")

    edge_predictor.to(config.device)
    edge_predictor.eval()

    for data in tqdm(dataset, desc="Robust augmenting with edge predictor"):
        try:
            # 🔧 修复：处理缺失protein_context的情况
            if hasattr(data, 'protein_context'):
                protein_context = data.protein_context.to(config.device)
            else:
                protein_context = data.x.to(config.device)

            # 提取正样本用于质量评估
            pos_mask = (data.y == 1)
            if pos_mask.sum() == 0:
                augmented_data.append(data)
                continue

            real_pos_samples = data.x[pos_mask].cpu().numpy()
            n_pos = pos_mask.sum().item()
            n_neg = (data.y == 0).sum().item()
            total_nodes = n_pos + n_neg

            # 计算生成数量
            target_pos = int(total_nodes * config.target_ratio)
            n_to_generate = max(config.min_samples_per_protein, target_pos - n_pos)
            n_to_generate = min(n_to_generate, int(n_pos * config.max_augment_ratio))

            if n_to_generate > 0:
                # 生成候选样本
                candidate_samples = diffusion_model.generate_positive_sample(
                    protein_context,
                    num_samples=n_to_generate * 3,
                    verbose=config.verbose_loading
                )

                if candidate_samples is None or len(candidate_samples) == 0:
                    augmented_data.append(data)
                    continue

                # 质量控制
                quality_samples, quality_score = calculate_sample_quality(
                    candidate_samples, real_pos_samples, config.quality_threshold
                )

                # 多样性控制
                if len(quality_samples) > 0:
                    diverse_samples, diversity_score = calculate_sample_diversity(
                        quality_samples, config.diversity_threshold
                    )
                else:
                    diverse_samples, diversity_score = candidate_samples[:n_to_generate], 0.5

                final_samples = diverse_samples[:n_to_generate]

                quality_stats.append(quality_score)
                diversity_stats.append(diversity_score)

                if len(final_samples) > 0:
                    # 使用边预测器构建边
                    new_x = torch.tensor(final_samples, dtype=torch.float32)

                    augmented_graph, edge_info = build_edges_with_edge_predictor(
                        data,
                        new_x,
                        edge_predictor,
                        config.device,
                        **predictor_config,
                        verbose=config.verbose_loading
                    )

                    edge_stats.append(edge_info)

                    # 限制图大小
                    if len(augmented_graph.x) > config.max_nodes_per_graph:
                        pos_mask_new = (augmented_graph.y == 1)
                        neg_mask_new = (augmented_graph.y == 0)

                        pos_indices = torch.where(pos_mask_new)[0]
                        neg_indices = torch.where(neg_mask_new)[0]

                        max_neg = config.max_nodes_per_graph - len(pos_indices)
                        if max_neg > 0 and len(neg_indices) > max_neg:
                            keep_neg = neg_indices[torch.randperm(len(neg_indices))[:max_neg]]
                            keep_indices = torch.cat([pos_indices, keep_neg])
                        else:
                            keep_indices = torch.arange(len(augmented_graph.x))

                        augmented_graph.x = augmented_graph.x[keep_indices]
                        augmented_graph.y = augmented_graph.y[keep_indices]

                        # 🔧 修复：更新边索引时也需要做边界检查
                        if augmented_graph.edge_index.numel() > 0:
                            mask = torch.isin(augmented_graph.edge_index[0], keep_indices) & \
                                   torch.isin(augmented_graph.edge_index[1], keep_indices)
                            augmented_graph.edge_index = augmented_graph.edge_index[:, mask]

                    # 保存蛋白质上下文信息
                    if hasattr(data, 'protein_context'):
                        augmented_graph.protein_context = data.protein_context
                    if hasattr(data, 'name'):
                        augmented_graph.name = data.name + "_aug_edgepred"

                    augmented_data.append(augmented_graph)
                else:
                    augmented_data.append(data)
            else:
                augmented_data.append(data)

        except Exception as e:
            print(f"⚠️  Warning: Augmentation failed for {getattr(data, 'name', 'unknown')}: {e}")
            augmented_data.append(data)

    # 统计信息
    if quality_stats:
        avg_quality = np.mean(quality_stats)
        avg_diversity = np.mean(diversity_stats)
        print(f"✅ 增强质量: 平均质量={avg_quality:.3f}, 平均多样性={avg_diversity:.3f}")

    if edge_stats:
        total_new_edges = sum([e['num_new_edges'] for e in edge_stats])
        avg_score = np.mean([np.mean(e['edge_scores']) for e in edge_stats if e['edge_scores']])
        print(f"✅ 边预测统计: 总新增边数={total_new_edges}, 平均边分数={avg_score:.3f}")

    return augmented_data, {
        'quality_scores': quality_stats,
        'diversity_scores': diversity_stats,
        'edge_stats': edge_stats
    }


def calculate_sample_quality(generated_samples, real_samples, threshold=0.7):
    """评估生成样本质量"""
    if len(generated_samples) == 0 or len(real_samples) == 0:
        return [], 0.0

    try:
        distances = pairwise_distances(generated_samples, real_samples, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        max_dist = np.max(min_distances) + 1e-8
        quality_scores = 1.0 - (min_distances / max_dist)

        high_quality_mask = quality_scores >= threshold
        high_quality_samples = generated_samples[high_quality_mask]
        avg_quality = np.mean(quality_scores)

        return high_quality_samples, avg_quality
    except:
        return generated_samples, 0.5


def calculate_sample_diversity(samples, threshold=0.3):
    """评估样本多样性"""
    if len(samples) <= 1:
        return samples, 1.0

    try:
        distances = pairwise_distances(samples, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        diverse_indices = []
        for i in range(len(samples)):
            min_dist = np.min(distances[i])
            if len(diverse_indices) == 0 or min_dist >= threshold:
                diverse_indices.append(i)

        diverse_samples = samples[diverse_indices]
        diversity_score = len(diverse_samples) / len(samples)

        return diverse_samples, diversity_score
    except:
        return samples, 1.0
