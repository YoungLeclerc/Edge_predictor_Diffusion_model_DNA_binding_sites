#!/usr/bin/env python3
"""
方案2改进：生成大规模真实格式的PPI数据
真实的PPI数据库结构和特性

这个脚本生成的数据具有真实特征：
- 遵循无尺度网络的幂律分布（真实生物网络的特性）
- 包含聚类系数（真实生物网络的特性）
- 数据量大（数万级别的相互作用对）
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import PPI_RAW_DIR


class RealisticPPIGenerator:
    """真实格式PPI数据生成器"""

    def __init__(self, num_proteins=5000, avg_degree=8, seed=42):
        """
        初始化

        Args:
            num_proteins: 蛋白质数量
            avg_degree: 平均度数 (每个蛋白质平均相互作用数)
            seed: 随机种子
        """
        self.num_proteins = num_proteins
        self.avg_degree = avg_degree
        self.seed = seed
        np.random.seed(seed)

    def generate_protein_names(self):
        """生成蛋白质名称"""
        print("🧬 生成蛋白质名称...")

        # 使用真实的通用蛋白质命名前缀
        prefixes = ['YP_', 'NP_', 'WP_', 'ZP_', 'AP_', 'XP_', 'RP_']
        names = []

        for i in range(self.num_proteins):
            prefix = np.random.choice(prefixes)
            number = str(np.random.randint(100000, 9999999))
            names.append(f"{prefix}{number}.1")

        print(f"✅ 生成 {len(names)} 个蛋白质名称")
        return names

    def generate_ppi_network(self, protein_names):
        """生成PPI网络 - 模拟无尺度网络"""
        print("\n🔗 生成PPI网络...")
        print(f"   • 蛋白质数: {self.num_proteins}")
        print(f"   • 平均度数: {self.avg_degree}")

        edges = []
        edge_set = set()

        # 使用优先连接（Preferential Attachment）模型生成无尺度网络
        # 这模拟了真实生物网络的特性
        degrees = [0] * self.num_proteins

        # 第一阶段：初始化小图
        for i in range(3):
            for j in range(i+1, 3):
                edges.append([i, j])
                edges.append([j, i])
                degrees[i] += 1
                degrees[j] += 1

        # 第二阶段：添加新节点并优先连接到高度数节点
        for new_node in range(3, self.num_proteins):
            # 选择要连接的节点
            num_connections = np.random.poisson(max(1, self.avg_degree / 2))
            num_connections = min(num_connections, new_node)

            # 根据度数进行优先连接（Barabási–Albert模型）
            probabilities = np.array(degrees[:new_node]) + 1.0
            probabilities = probabilities / probabilities.sum()

            selected_nodes = np.random.choice(
                range(new_node),
                size=num_connections,
                replace=False,
                p=probabilities
            )

            for node in selected_nodes:
                edges.append([new_node, node])
                edges.append([node, new_node])
                degrees[new_node] += 1
                degrees[node] += 1

        # 添加随机边增加聚类系数
        num_random_edges = int(len(edges) * 0.1)  # 额外10%的随机边
        for _ in range(num_random_edges):
            i = np.random.randint(0, self.num_proteins)
            j = np.random.randint(0, self.num_proteins)

            if i != j and (i, j) not in edge_set:
                edges.append([i, j])
                edge_set.add((i, j))

        print(f"✅ 生成 {len(edges)} 条边")

        # 统计
        print(f"\n📊 网络统计:")
        print(f"   • 度数范围: [{min(degrees)}, {max(degrees)}]")
        print(f"   • 平均度数: {np.mean(degrees):.2f}")
        print(f"   • 度数标准差: {np.std(degrees):.2f}")

        return edges, degrees

    def generate_ppi_data(self, edges, protein_names):
        """生成PPI数据框"""
        print("\n📝 生成PPI数据框...")

        # 真实的实验方法类型
        experimental_systems = [
            'Two-hybrid',
            'Affinity Capture-MS',
            'Affinity Capture-Western',
            'Biochemical Activity',
            'Co-immunoprecipitation',
            'Co-localization',
            'Protein-peptide',
            'Reconstituted Complex',
            'FRAP',
            'Yeast 2-hybrid',
        ]

        # 生成数据
        ppi_data = []
        seen_pairs = set()

        for idx, (src, dst) in enumerate(tqdm(edges, desc="生成相互作用")):
            # 避免自循环
            if src == dst:
                continue

            # 避免重复（只保留src < dst的)
            if src > dst:
                src, dst = dst, src

            if (src, dst) not in seen_pairs:
                seen_pairs.add((src, dst))
                ppi_data.append({
                    'BioGRID Interaction ID': str(1000000 + idx),
                    'Official Symbol Interactor A': protein_names[src],
                    'Official Symbol Interactor B': protein_names[dst],
                    'Experimental System': np.random.choice(experimental_systems),
                    'Throughput': np.random.choice(['Low Throughput', 'High Throughput']),
                    'Score': np.random.uniform(0.4, 1.0) if np.random.random() < 0.7 else None,
                    'Pubmed ID': str(np.random.randint(10000000, 99999999)),
                })

        df = pd.DataFrame(ppi_data)

        print(f"✅ 生成 {len(df)} 条有效的相互作用记录")

        return df

    def save_data(self, ppi_df, protein_names, degrees):
        """保存数据"""
        print("\n💾 保存数据...")

        os.makedirs(PPI_RAW_DIR, exist_ok=True)

        # 保存PPI数据
        ppi_file = os.path.join(PPI_RAW_DIR, "ppi_data.csv")
        ppi_df.to_csv(ppi_file, index=False)
        print(f"✅ PPI数据: {ppi_file}")

        # 保存蛋白质信息
        protein_file = os.path.join(PPI_RAW_DIR, "proteins.csv")
        protein_df = pd.DataFrame({
            'protein_id': protein_names,
            'degree': degrees,
            'organism': ['Homo sapiens'] * len(protein_names)
        })
        protein_df.to_csv(protein_file, index=False)
        print(f"✅ 蛋白质信息: {protein_file}")

        return ppi_file

    def print_statistics(self, ppi_df, degrees):
        """打印统计信息"""
        print("\n📊 数据统计信息:")
        print(f"   • 总相互作用数: {len(ppi_df)}")
        print(f"   • 独特蛋白质对: {len(set(ppi_df['Official Symbol Interactor A']) | set(ppi_df['Official Symbol Interactor B']))}")

        if 'Experimental System' in ppi_df.columns:
            print(f"\n   实验方法分布:")
            exp_counts = ppi_df['Experimental System'].value_counts()
            for exp, count in exp_counts.head(5).items():
                print(f"      - {exp}: {count} ({100*count/len(ppi_df):.1f}%)")

        if 'Throughput' in ppi_df.columns:
            print(f"\n   高通量vs低通量:")
            throughput_counts = ppi_df['Throughput'].value_counts()
            for method, count in throughput_counts.items():
                print(f"      - {method}: {count} ({100*count/len(ppi_df):.1f}%)")

        print(f"\n   度数分布:")
        print(f"      - 最小: {min(degrees)}")
        print(f"      - 最大: {max(degrees)}")
        print(f"      - 平均: {np.mean(degrees):.2f}")
        print(f"      - 中位数: {np.median(degrees):.2f}")


def main():
    """主函数"""
    print("🧬 生成真实格式PPI数据")
    print("=" * 60)

    # 参数配置
    NUM_PROTEINS = 5000        # 蛋白质数量
    AVG_DEGREE = 8             # 平均度数
    SEED = 42

    print(f"⚙️  配置:")
    print(f"   • 蛋白质数: {NUM_PROTEINS}")
    print(f"   • 平均度数: {AVG_DEGREE}")
    print(f"   • 预期边数: ~{NUM_PROTEINS * AVG_DEGREE // 2}")

    # 生成数据
    generator = RealisticPPIGenerator(
        num_proteins=NUM_PROTEINS,
        avg_degree=AVG_DEGREE,
        seed=SEED
    )

    # 第1步：生成蛋白质名称
    protein_names = generator.generate_protein_names()

    # 第2步：生成网络
    edges, degrees = generator.generate_ppi_network(protein_names)

    # 第3步：生成PPI数据
    ppi_df = generator.generate_ppi_data(edges, protein_names)

    # 第4步：保存数据
    ppi_file = generator.save_data(ppi_df, protein_names, degrees)

    # 第5步：打印统计
    generator.print_statistics(ppi_df, degrees)

    print("\n" + "=" * 60)
    print("✅ 步骤1完成: PPI数据已生成")
    print("=" * 60)
    print("\n📋 特点:")
    print("   ✨ 遵循无尺度网络分布 (真实生物网络特性)")
    print("   ✨ 包含多种实验方法类型")
    print("   ✨ 包含高通量和低通量实验混合")
    print("   ✨ 数据量大 (数万级边)")
    print("\n👉 下一步:")
    print("   python 2_preprocess_ppi.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
