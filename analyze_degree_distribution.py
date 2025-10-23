#!/usr/bin/env python3
"""
分析PPI网络的度数分布
验证负样本采样是否真正无偏差
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# 配置
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")
PPI_PROCESSED_DIR = os.path.join(DATA_DIR, "ppi_processed")


def analyze_network():
    """分析网络的度数分布"""
    print("🔍 分析PPI网络度数分布")
    print("=" * 70)

    # 加载PPI数据
    print("\n📊 加载数据...")
    ppi_file = os.path.join(PPI_RAW_DIR, "ppi_data.csv")
    ppi_data = pd.read_csv(ppi_file)

    # 加载蛋白质映射
    with open(os.path.join(PPI_PROCESSED_DIR, "protein_mapping.json"), 'r') as f:
        mapping = json.load(f)
        protein_to_idx = mapping['protein_to_idx']

    num_proteins = len(protein_to_idx)
    print(f"✅ 蛋白质总数: {num_proteins:,}")
    print(f"✅ 相互作用总数: {len(ppi_data):,}")

    # 计算每个蛋白质的度数
    print("\n📈 计算度数分布...")
    degree = defaultdict(int)

    for _, row in tqdm(ppi_data.iterrows(), total=len(ppi_data), desc="计算度数"):
        p1 = row['protein1']
        p2 = row['protein2']

        if p1 in protein_to_idx and p2 in protein_to_idx:
            idx1 = protein_to_idx[p1]
            idx2 = protein_to_idx[p2]

            if idx1 != idx2:
                degree[idx1] += 1
                degree[idx2] += 1

    # 统计度数分布
    degrees = [degree.get(i, 0) for i in range(num_proteins)]
    degrees_array = np.array(degrees)

    print("\n📊 度数统计:")
    print(f"   • 最小度数: {degrees_array.min()}")
    print(f"   • 最大度数: {degrees_array.max()}")
    print(f"   • 平均度数: {degrees_array.mean():.2f}")
    print(f"   • 中位数: {np.median(degrees_array):.2f}")
    print(f"   • 标准差: {degrees_array.std():.2f}")

    # 度数分布统计
    print("\n📊 度数分组统计:")
    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]
    for i in range(len(bins) - 1):
        count = np.sum((degrees_array > bins[i]) & (degrees_array <= bins[i+1]))
        percentage = 100 * count / num_proteins
        print(f"   • 度数 {bins[i]+1:4d} - {bins[i+1]:4d}: {count:6,} ({percentage:5.2f}%)")

    # 特别关注低度数蛋白
    print("\n🔍 低度数蛋白分析:")
    deg_0 = np.sum(degrees_array == 0)
    deg_1_5 = np.sum((degrees_array >= 1) & (degrees_array <= 5))
    deg_1_10 = np.sum((degrees_array >= 1) & (degrees_array <= 10))

    print(f"   • 度数 = 0（孤立节点）: {deg_0:,} ({100*deg_0/num_proteins:.2f}%)")
    print(f"   • 度数 1-5: {deg_1_5:,} ({100*deg_1_5/num_proteins:.2f}%)")
    print(f"   • 度数 1-10: {deg_1_10:,} ({100*deg_1_10/num_proteins:.2f}%)")

    # 理论计算：均匀采样时，涉及低度数蛋白的负样本比例
    print("\n🧮 理论分析（均匀采样）:")
    prob_low_degree = deg_1_5 / num_proteins  # 单个蛋白是低度数的概率
    # 一条边涉及低度数蛋白的概率（至少一端是低度数）
    prob_edge_with_low = 1 - (1 - prob_low_degree) ** 2

    print(f"   • 单个蛋白是低度数(≤5)的概率: {100*prob_low_degree:.2f}%")
    print(f"   • 理论上，边涉及低度数蛋白的概率: {100*prob_edge_with_low:.2f}%")

    # 加载实际的负样本
    print("\n📊 实际负样本分析:")
    edges_train = np.load(os.path.join(PPI_PROCESSED_DIR, "edges_train.npy"))
    labels_train = np.load(os.path.join(PPI_PROCESSED_DIR, "labels_train.npy"))

    negative_edges = edges_train[labels_train == 0]

    neg_degrees = []
    low_degree_count = 0
    for src, dst in negative_edges:
        deg_src = degree.get(src, 0)
        deg_dst = degree.get(dst, 0)
        neg_degrees.append(deg_src)
        neg_degrees.append(deg_dst)

        if deg_src <= 5 or deg_dst <= 5:
            low_degree_count += 1

    actual_percentage = 100 * low_degree_count / len(negative_edges)
    print(f"   • 实际涉及低度数蛋白(≤5)的负样本: {low_degree_count:,} ({actual_percentage:.2f}%)")
    print(f"   • 理论预期: {100*prob_edge_with_low:.2f}%")

    if abs(actual_percentage - 100*prob_edge_with_low) < 2:
        print(f"   ✅ 采样无偏差！实际比例接近理论预期")
    else:
        print(f"   ⚠️  采样可能有轻微偏差（差异: {abs(actual_percentage - 100*prob_edge_with_low):.2f}%）")

    # 绘制度数分布图
    print("\n📊 生成度数分布可视化...")
    plt.figure(figsize=(12, 5))

    # 左图：度数分布直方图（对数刻度）
    plt.subplot(1, 2, 1)
    plt.hist(degrees_array[degrees_array > 0], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Number of Proteins', fontsize=12)
    plt.title('Degree Distribution (Log Scale)', fontsize=14)
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # 右图：累积分布
    plt.subplot(1, 2, 2)
    sorted_degrees = np.sort(degrees_array)
    cumulative = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)
    plt.plot(sorted_degrees, cumulative, linewidth=2)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Degree Distribution', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xscale('log')

    plt.tight_layout()
    output_file = os.path.join(current_dir, "degree_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化已保存: {output_file}")

    # 保存度数数据
    degree_data = {
        'protein_idx': list(range(num_proteins)),
        'degree': [degree.get(i, 0) for i in range(num_proteins)]
    }
    degree_df = pd.DataFrame(degree_data)
    degree_df.to_csv(os.path.join(current_dir, "protein_degrees.csv"), index=False)
    print(f"✅ 度数数据已保存: protein_degrees.csv")

    print("\n" + "=" * 70)
    print("✅ 分析完成")


if __name__ == "__main__":
    analyze_network()
