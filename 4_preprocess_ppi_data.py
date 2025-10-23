#!/usr/bin/env python3
"""
步骤4: 预处理真实的PPI数据（最终版）
加载ESM2特征和PPI数据，生成训练集
改进：无偏差的负样本生成，提供多种采样策略
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")
PPI_PROCESSED_DIR = os.path.join(DATA_DIR, "ppi_processed")


class RealPPIPreprocessor:
    """真实PPI数据预处理器"""

    def __init__(self):
        self.features = None
        self.protein_to_idx = {}
        self.idx_to_protein = {}
        self.ppi_data = None

    def load_esm2_features(self):
        """加载ESM2特征"""
        print("\n📊 加载ESM2特征...")

        features_file = os.path.join(PPI_PROCESSED_DIR, "features.npy")
        mapping_file = os.path.join(PPI_PROCESSED_DIR, "protein_mapping.json")

        if not os.path.exists(features_file):
            print(f"❌ 错误: ESM2特征文件不存在: {features_file}")
            print("请先运行: python 3_extract_esm2_features.py")
            return False

        # 加载特征
        self.features = np.load(features_file)
        print(f"✅ 特征已加载: {self.features.shape}")

        # 加载映射
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            self.protein_to_idx = mapping['protein_to_idx']
            self.idx_to_protein = {int(k): v for k, v in mapping['idx_to_protein'].items()}

        print(f"✅ 蛋白质映射已加载: {len(self.protein_to_idx)} 个蛋白质")
        print(f"\n📊 特征统计:")
        print(f"   • 均值: {self.features.mean():.6f}")
        print(f"   • 标准差: {self.features.std():.6f}")
        print(f"   • 维度: {self.features.shape[1]}")

        return True

    def load_ppi_data(self):
        """加载真实PPI数据"""
        print("\n🔗 加载PPI数据...")

        ppi_file = os.path.join(PPI_RAW_DIR, "ppi_data.csv")
        if not os.path.exists(ppi_file):
            print(f"❌ 错误: PPI数据文件不存在: {ppi_file}")
            return False

        self.ppi_data = pd.read_csv(ppi_file)
        print(f"✅ PPI数据已加载: {len(self.ppi_data):,} 条相互作用")

        return True

    def extract_edges(self):
        """提取正样本边"""
        print("\n🔗 提取正样本边...")

        positive_edges = []
        proteins_with_features = set(self.protein_to_idx.keys())

        for _, row in tqdm(self.ppi_data.iterrows(), total=len(self.ppi_data), desc="提取边"):
            p1 = row['protein1']
            p2 = row['protein2']

            # 检查两个蛋白质是否都有ESM2特征
            if p1 in proteins_with_features and p2 in proteins_with_features:
                src_idx = self.protein_to_idx[p1]
                dst_idx = self.protein_to_idx[p2]

                # 避免自环
                if src_idx != dst_idx:
                    # 规范化：小索引在前
                    edge = (min(src_idx, dst_idx), max(src_idx, dst_idx))
                    positive_edges.append(edge)

        # 去重
        positive_edges = list(set(positive_edges))
        positive_edges = np.array(positive_edges, dtype=np.int32)

        print(f"✅ 正样本边: {len(positive_edges):,}")
        print(f"   覆盖蛋白质: {len(set(positive_edges.flatten())):,}")

        return positive_edges

    def generate_negative_samples(self, positive_edges, num_proteins, ratio=1.0,
                                  strategy='uniform'):
        """
        生成负样本 - 支持多种采样策略

        参数:
            positive_edges: 正样本边
            num_proteins: 蛋白质总数
            ratio: 负样本/正样本比例 (默认1:1)
            strategy: 采样策略
                - 'uniform': 均匀随机采样（无偏差，推荐）
                - 'degree_aware': 度数感知采样（偏向hub蛋白）
                - 'mixed': 混合策略（50%均匀 + 50%度数感知）
        """
        print(f"\n❌ 生成负样本 (比例 {ratio}:1, 策略: {strategy})...")

        # 构建正样本集合（快速查询）
        positive_set = set()
        degree = defaultdict(int)

        for src, dst in positive_edges:
            src, dst = int(src), int(dst)
            edge = (min(src, dst), max(src, dst))
            positive_set.add(edge)
            degree[src] += 1
            degree[dst] += 1

        target_negs = int(len(positive_edges) * ratio)
        print(f"   目标负样本数: {target_negs:,}")

        # 计算网络统计
        max_edges = num_proteins * (num_proteins - 1) // 2
        density = len(positive_set) / max_edges
        print(f"   网络密度: {density:.6f}")

        # 度数分布统计
        degrees_list = list(degree.values())
        if degrees_list:
            print(f"   度数统计: 最小={min(degrees_list)}, 最大={max(degrees_list)}, "
                  f"平均={np.mean(degrees_list):.1f}, 中位数={np.median(degrees_list):.1f}")

        # 选择采样策略
        if strategy == 'uniform':
            negative_edges = self._sample_uniform(positive_set, num_proteins, target_negs)
        elif strategy == 'degree_aware':
            negative_edges = self._sample_degree_aware(positive_set, degree, num_proteins, target_negs)
        elif strategy == 'mixed':
            # 50% 均匀 + 50% 度数感知
            half = target_negs // 2
            print(f"   混合策略: {half:,} 均匀 + {target_negs - half:,} 度数感知")
            neg1 = self._sample_uniform(positive_set, num_proteins, half)
            neg2 = self._sample_degree_aware(positive_set, degree, num_proteins, target_negs - half)
            negative_edges = list(set(neg1) | set(neg2))
        else:
            raise ValueError(f"未知的采样策略: {strategy}")

        negative_edges = np.array(negative_edges, dtype=np.int32)

        print(f"✅ 负样本边: {len(negative_edges):,}")

        # 分析负样本的度数分布
        self._analyze_negative_samples(negative_edges, degree)

        return negative_edges

    def _sample_uniform(self, positive_set, num_proteins, target_negs):
        """
        均匀随机采样（无偏差）

        优点：
        - 无偏差，所有蛋白质被选中的概率相同
        - 包含"困难负样本"（低度数蛋白之间的边）
        - 更好地反映真实的负样本分布

        适用于：大多数情况（推荐作为默认策略）
        """
        negative_edges = set()
        batch_size = min(target_negs * 2, 1000000)

        with tqdm(total=target_negs, desc="均匀采样") as pbar:
            while len(negative_edges) < target_negs:
                # 批量生成候选（均匀分布）
                src = np.random.randint(0, num_proteins, size=batch_size)
                dst = np.random.randint(0, num_proteins, size=batch_size)

                # 过滤：去除自环
                mask = src != dst
                src = src[mask]
                dst = dst[mask]

                # 规范化并去重
                candidates = set()
                for s, d in zip(src, dst):
                    edge = (min(s, d), max(s, d))
                    if edge not in positive_set and edge not in negative_edges:
                        candidates.add(edge)

                # 添加到负样本集
                needed = target_negs - len(negative_edges)
                to_add = list(candidates)[:needed]
                negative_edges.update(to_add)
                pbar.update(len(to_add))

        return list(negative_edges)

    def _sample_degree_aware(self, positive_set, degree, num_proteins, target_negs):
        """
        度数感知采样（有偏差）

        特点：
        - 高度数蛋白（hub）更容易被选中
        - 负样本偏向于hub蛋白之间的非连接边
        - 可能过度拟合到度数特征

        适用于：当你想让模型学习度数信息时
        """
        # 计算采样概率（基于度数）
        degrees = np.array([degree.get(i, 0) + 1 for i in range(num_proteins)])
        prob = degrees / degrees.sum()

        negative_edges = set()
        batch_size = min(target_negs * 3, 1000000)

        with tqdm(total=target_negs, desc="度数感知采样") as pbar:
            attempts = 0
            max_attempts = target_negs * 20

            while len(negative_edges) < target_negs and attempts < max_attempts:
                # 批量采样（按度数加权）
                src = np.random.choice(num_proteins, size=batch_size, p=prob)
                dst = np.random.choice(num_proteins, size=batch_size, p=prob)

                # 过滤自环
                mask = src != dst
                src = src[mask]
                dst = dst[mask]

                # 规范化并去重
                candidates = set()
                for s, d in zip(src, dst):
                    edge = (min(s, d), max(s, d))
                    if edge not in positive_set and edge not in negative_edges:
                        candidates.add(edge)

                # 添加
                needed = target_negs - len(negative_edges)
                to_add = list(candidates)[:needed]
                negative_edges.update(to_add)
                pbar.update(len(to_add))

                attempts += 1

        if len(negative_edges) < target_negs:
            print(f"   ⚠️  警告: 仅生成了 {len(negative_edges):,}/{target_negs:,} 个负样本")

        return list(negative_edges)

    def _analyze_negative_samples(self, negative_edges, degree):
        """分析负样本的度数分布"""
        neg_degrees = []
        for src, dst in negative_edges:
            neg_degrees.append(degree.get(src, 0))
            neg_degrees.append(degree.get(dst, 0))

        if neg_degrees:
            print(f"   负样本度数分布: 最小={min(neg_degrees)}, 最大={max(neg_degrees)}, "
                  f"平均={np.mean(neg_degrees):.1f}, 中位数={np.median(neg_degrees):.1f}")

            # 统计涉及低度数蛋白的负样本
            low_degree_edges = sum(1 for src, dst in negative_edges
                                  if degree.get(src, 0) <= 5 or degree.get(dst, 0) <= 5)
            print(f"   涉及低度数蛋白(≤5)的负样本: {low_degree_edges:,} "
                  f"({100*low_degree_edges/len(negative_edges):.1f}%)")

    def create_train_val_test_split(self, positive_edges, negative_edges,
                                    train_ratio=0.8, val_ratio=0.1):
        """创建训练/验证/测试集"""
        print("\n📊 创建数据划分...")

        # 合并边和标签
        all_edges = np.vstack([positive_edges, negative_edges])
        all_labels = np.array(
            [1] * len(positive_edges) + [0] * len(negative_edges),
            dtype=np.int32
        )

        # 随机打乱
        np.random.seed(42)
        indices = np.random.permutation(len(all_edges))
        all_edges = all_edges[indices]
        all_labels = all_labels[indices]

        # 分割
        n_total = len(all_edges)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        edges_train = all_edges[:n_train]
        labels_train = all_labels[:n_train]

        edges_val = all_edges[n_train:n_train+n_val]
        labels_val = all_labels[n_train:n_train+n_val]

        edges_test = all_edges[n_train+n_val:]
        labels_test = all_labels[n_train+n_val:]

        print(f"✅ 数据划分完成:")
        print(f"   • 训练集: {len(edges_train):,} ({100*len(edges_train)/n_total:.1f}%)")
        print(f"     └─ 正样本: {(labels_train == 1).sum():,}, 负样本: {(labels_train == 0).sum():,}")
        print(f"   • 验证集: {len(edges_val):,} ({100*len(edges_val)/n_total:.1f}%)")
        print(f"     └─ 正样本: {(labels_val == 1).sum():,}, 负样本: {(labels_val == 0).sum():,}")
        print(f"   • 测试集: {len(edges_test):,} ({100*len(edges_test)/n_total:.1f}%)")
        print(f"     └─ 正样本: {(labels_test == 1).sum():,}, 负样本: {(labels_test == 0).sum():,}")

        return (edges_train, labels_train,
                edges_val, labels_val,
                edges_test, labels_test)

    def save_processed_data(self, edges_train, labels_train,
                           edges_val, labels_val,
                           edges_test, labels_test,
                           strategy='uniform'):
        """保存预处理后的数据"""
        print("\n💾 保存处理后的数据...")

        # 保存边
        np.save(os.path.join(PPI_PROCESSED_DIR, "edges_train.npy"), edges_train)
        np.save(os.path.join(PPI_PROCESSED_DIR, "edges_val.npy"), edges_val)
        np.save(os.path.join(PPI_PROCESSED_DIR, "edges_test.npy"), edges_test)

        # 保存标签（重要！）
        np.save(os.path.join(PPI_PROCESSED_DIR, "labels_train.npy"), labels_train)
        np.save(os.path.join(PPI_PROCESSED_DIR, "labels_val.npy"), labels_val)
        np.save(os.path.join(PPI_PROCESSED_DIR, "labels_test.npy"), labels_test)

        print(f"✅ 已保存:")
        print(f"   • 训练边: edges_train.npy")
        print(f"   • 训练标签: labels_train.npy")
        print(f"   • 验证边: edges_val.npy")
        print(f"   • 验证标签: labels_val.npy")
        print(f"   • 测试边: edges_test.npy")
        print(f"   • 测试标签: labels_test.npy")

        # 保存元信息
        meta_info = {
            'num_proteins': len(self.protein_to_idx),
            'feature_dim': self.features.shape[1],
            'num_train': len(edges_train),
            'num_val': len(edges_val),
            'num_test': len(edges_test),
            'num_positive_train': int((labels_train == 1).sum()),
            'num_negative_train': int((labels_train == 0).sum()),
            'negative_sampling_strategy': strategy,
            'data_source': 'STRING v12.0 (Homo sapiens)',
            'features': 'ESM2 (facebook/esm2_t33_650M_UR50D)'
        }

        with open(os.path.join(PPI_PROCESSED_DIR, "meta_info.json"), 'w') as f:
            json.dump(meta_info, f, indent=2)

        print(f"   • 元信息: meta_info.json")


def main():
    print("🔗 步骤4: 预处理真实PPI数据")
    print("=" * 70)

    preprocessor = RealPPIPreprocessor()

    # 步骤1: 加载ESM2特征
    if not preprocessor.load_esm2_features():
        return False

    # 步骤2: 加载PPI数据
    if not preprocessor.load_ppi_data():
        return False

    # 步骤3: 提取正样本边
    positive_edges = preprocessor.extract_edges()

    # 步骤4: 生成负样本
    num_proteins = len(preprocessor.protein_to_idx)

    # 选择采样策略
    # 推荐: 'uniform' - 无偏差，更真实反映负样本分布
    # 备选: 'degree_aware' - 偏向hub蛋白，可能过拟合
    # 备选: 'mixed' - 混合策略
    SAMPLING_STRATEGY = 'uniform'  # 修改这里可以切换策略

    negative_edges = preprocessor.generate_negative_samples(
        positive_edges,
        num_proteins,
        ratio=1.0,  # 1:1比例
        strategy=SAMPLING_STRATEGY
    )

    # 步骤5: 创建划分
    result = preprocessor.create_train_val_test_split(
        positive_edges,
        negative_edges
    )
    edges_train, labels_train, edges_val, labels_val, edges_test, labels_test = result

    # 步骤6: 保存数据
    preprocessor.save_processed_data(
        edges_train, labels_train,
        edges_val, labels_val,
        edges_test, labels_test,
        strategy=SAMPLING_STRATEGY
    )

    print("\n" + "=" * 70)
    print("✅ 步骤4完成: 真实PPI数据已预处理")
    print(f"📁 数据位置: {PPI_PROCESSED_DIR}")
    print(f"\n💡 使用的负样本策略: {SAMPLING_STRATEGY}")
    print("   • uniform: 无偏差，推荐用于大多数情况")
    print("   • degree_aware: 偏向hub蛋白，可能导致过拟合")
    print("   • mixed: 50%均匀 + 50%度数感知")
    print("\n👉 下一步: 训练边预测器")
    print("   运行: python 5_train_edge_predictor.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
