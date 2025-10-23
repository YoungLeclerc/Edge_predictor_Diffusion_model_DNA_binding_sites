#!/usr/bin/env python3
"""
方案2步骤0: 下载完整BioGRID数据集
从BioGRID数据库下载完整的蛋白质相互作用数据
"""
import os
import sys
import urllib.request
import tarfile
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/biogrid_raw")
os.makedirs(DATA_DIR, exist_ok=True)

# BioGRID下载链接
BIOGRID_URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-Latest.tab.zip"
BIOGRID_FILE = os.path.join(DATA_DIR, "BIOGRID-ALL-Latest.tab.zip")
BIOGRID_EXTRACTED = os.path.join(DATA_DIR, "BIOGRID-ALL-Latest.tab")


def download_biogrid():
    """下载BioGRID数据"""
    print("\n🔗 下载BioGRID数据...")
    print(f"   📥 URL: {BIOGRID_URL}")
    print(f"   💾 保存位置: {BIOGRID_FILE}")

    if os.path.exists(BIOGRID_FILE):
        print(f"   ⚠️  文件已存在，跳过下载")
        return BIOGRID_FILE

    try:
        print("   ⏳ 下载中... (这可能需要几分钟)")
        urllib.request.urlretrieve(BIOGRID_URL, BIOGRID_FILE)
        print(f"   ✅ 下载完成: {BIOGRID_FILE}")
        return BIOGRID_FILE
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        print(f"\n   💡 提示: 你可以手动下载:")
        print(f"      1. 访问 https://thebiogrid.org/download.php")
        print(f"      2. 下载最新版本的 'BIOGRID-ALL-Latest.tab.zip'")
        print(f"      3. 放在 {DATA_DIR} 目录")
        return None


def extract_biogrid(zip_file):
    """解压BioGRID数据"""
    print("\n📦 解压BioGRID数据...")

    if not os.path.exists(zip_file):
        print(f"   ❌ 文件不存在: {zip_file}")
        return None

    try:
        with tarfile.open(zip_file) as tar:
            tar.extractall(DATA_DIR)
        print(f"   ✅ 解压完成")

        # 查找实际的数据文件
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith(".tab") and "BIOGRID-ALL" in file:
                    return os.path.join(root, file)
        return None
    except Exception as e:
        print(f"   ❌ 解压失败: {e}")
        return None


def analyze_biogrid_data(tab_file):
    """分析BioGRID数据"""
    print("\n📊 分析BioGRID数据...")
    print(f"   📖 读取文件: {tab_file}")

    try:
        # 读取BioGRID数据
        df = pd.read_csv(tab_file, sep='\t', dtype=str)
        print(f"   ✅ 数据加载完成")

        # 显示列信息
        print(f"\n   📋 数据列:")
        for i, col in enumerate(df.columns):
            print(f"      {i}: {col}")

        print(f"\n   📊 基本统计:")
        print(f"      • 总行数: {len(df)}")
        print(f"      • 蛋白质A列: {df.iloc[:, 4].nunique() if len(df.columns) > 4 else 'N/A'}")
        print(f"      • 蛋白质B列: {df.iloc[:, 5].nunique() if len(df.columns) > 5 else 'N/A'}")

        # 获取物种信息
        if len(df.columns) > 9:
            taxid_col = df.iloc[:, 9]
            species_counts = taxid_col.value_counts()
            print(f"\n   🧬 物种分布 (前10):")
            for species, count in species_counts.head(10).items():
                print(f"      • TaxID {species}: {count} 相互作用")

        return df
    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return None


def extract_species_data(df, target_taxid="559292"):
    """
    提取特定物种的数据
    559292 = S. cerevisiae (酵母，最常用的模型生物)
    9606 = Homo sapiens (人类，最大的数据集)
    """
    print(f"\n🧬 提取物种数据 (TaxID: {target_taxid})...")

    # BioGRID列的索引
    PROT_A_IDX = 4    # Official Symbol Interactor A
    PROT_B_IDX = 5    # Official Symbol Interactor B
    TAXID_IDX = 9     # Organism Interactor A (TaxID)
    EXP_SYS_IDX = 6   # Experimental System

    # 过滤指定物种
    if len(df.columns) > TAXID_IDX:
        species_df = df[df.iloc[:, TAXID_IDX] == target_taxid].copy()
    else:
        # 如果没有TaxID列，使用所有数据
        species_df = df.copy()

    print(f"   ✅ 提取完成: {len(species_df)} 条相互作用")

    if len(species_df) == 0:
        print(f"   ⚠️  未找到 TaxID={target_taxid} 的数据，使用所有数据")
        species_df = df

    # 统计信息
    if len(species_df.columns) > PROT_A_IDX:
        proteins = set(species_df.iloc[:, PROT_A_IDX]) | set(species_df.iloc[:, PROT_B_IDX])
        print(f"   • 蛋白质数: {len(proteins)}")
        print(f"   • 相互作用数: {len(species_df)}")

    return species_df


def generate_summary():
    """生成使用说明"""
    print("\n" + "=" * 70)
    print("📋 BioGRID数据下载说明")
    print("=" * 70)

    print("""
🎯 下一步:

1️⃣  自动下载方式 (推荐):
   • 确保网络连接正常
   • 脚本会自动下载 BioGRID 最新版本
   • 文件大小: ~100-300 MB (取决于版本)
   • 下载时间: 5-15 分钟

2️⃣  手动下载方式:
   如果自动下载失败，请:
   • 访问 https://thebiogrid.org/download.php
   • 下载最新版本的 'BIOGRID-ALL-Latest.tab.zip'
   • 放在: {0}
   • 然后重新运行此脚本

3️⃣  数据说明:
   • 格式: BioGRID TAB-delimited
   • 来源: 已发表的蛋白质相互作用
   • 物种: 多物种 (~30+)
   • 最大数据集: 人类 (9606) 或酵母 (559292)

4️⃣  处理步骤:
   ✅ 下载完整数据
   ✅ 提取最大物种 (人类或酵母)
   ✅ 清理和验证
   ✅ 生成特征
   ✅ 训练新模型

📊 预期结果:
   • 蛋白质数: 15,000 - 40,000
   • 相互作用: 200,000 - 500,000
   • 训练样本: 400,000 - 1,000,000
   • 模型性能: AUC > 0.85
    """.format(DATA_DIR))


if __name__ == "__main__":
    print("🔗 方案2步骤0: 下载完整BioGRID数据")
    print("=" * 70)

    # 第1步：下载
    zip_file = download_biogrid()

    if zip_file:
        # 第2步：解压
        tab_file = extract_biogrid(zip_file)

        if tab_file and os.path.exists(tab_file):
            # 第3步：分析
            df = analyze_biogrid_data(tab_file)

            if df is not None:
                # 第4步：提取物种数据
                print("\n🧬 尝试提取多个物种数据...")

                # 尝试提取人类数据
                human_data = extract_species_data(df, target_taxid="9606")

                if len(human_data) > len(df) * 0.1:  # 如果人类数据较大
                    print("   → 选择人类数据集")
                    species_data = human_data
                else:
                    # 尝试提取酵母数据
                    yeast_data = extract_species_data(df, target_taxid="559292")
                    if len(yeast_data) > len(df) * 0.1:
                        print("   → 选择酵母数据集")
                        species_data = yeast_data
                    else:
                        print("   → 使用所有物种数据")
                        species_data = df

                # 保存物种数据
                output_file = os.path.join(DATA_DIR, "biogrid_species_interactions.csv")
                species_data.to_csv(output_file, sep=',', index=False)
                print(f"\n💾 数据已保存: {output_file}")
                print(f"   • 大小: {os.path.getsize(output_file) / (1024**2):.2f} MB")

        # 第5步：生成说明
        generate_summary()

    print("\n👉 下一步: 运行 1_preprocess_biogrid_data.py")
