#!/usr/bin/env python3
"""
下载真实的人类PPI数据
数据源: STRING数据库 (人类蛋白质相互作用)
"""
import os
import sys
import pandas as pd
import requests
from tqdm import tqdm

# 导入配置
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "ppi_config.py")
spec = importlib.util.spec_from_file_location("ppi_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

DATA_DIR = os.path.join(current_dir, "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")
os.makedirs(PPI_RAW_DIR, exist_ok=True)


def download_string_ppi():
    """
    下载STRING人类PPI数据
    STRING提供高质量的蛋白质相互作用数据
    物种ID: 9606 (人类)
    """
    print("🔗 下载STRING人类PPI数据")
    print("=" * 70)

    # STRING数据库链接 (人类 - NCBI Taxonomy ID: 9606)
    # 使用物理相互作用 (physical interactions)，置信度 > 400 (中等置信度)
    string_url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"

    output_file = os.path.join(PPI_RAW_DIR, "9606.protein.links.v12.0.txt.gz")

    print(f"📥 下载URL: {string_url}")
    print(f"💾 保存位置: {output_file}")

    try:
        # 下载文件
        response = requests.get(string_url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ 下载完成: {output_file}")

        # 解压并处理
        print("\n📦 解压并处理数据...")
        import gzip

        uncompressed_file = output_file.replace('.gz', '')

        with gzip.open(output_file, 'rb') as f_in:
            with open(uncompressed_file, 'wb') as f_out:
                f_out.write(f_in.read())

        print(f"✅ 解压完成: {uncompressed_file}")

        # 读取并过滤数据
        print("\n📊 加载并过滤数据 (置信度 >= 400)...")
        df = pd.read_csv(uncompressed_file, sep=' ')

        print(f"   原始数据: {len(df):,} 条相互作用")

        # 过滤低置信度的相互作用 (combined_score >= 400)
        df_filtered = df[df['combined_score'] >= 400].copy()

        print(f"   过滤后: {len(df_filtered):,} 条相互作用 (combined_score >= 400)")

        # 提取蛋白质名称 (去掉物种前缀 "9606.")
        df_filtered['protein1'] = df_filtered['protein1'].str.replace('9606.', '', regex=False)
        df_filtered['protein2'] = df_filtered['protein2'].str.replace('9606.', '', regex=False)

        # 保存处理后的数据
        output_csv = os.path.join(PPI_RAW_DIR, "ppi_data.csv")
        df_filtered[['protein1', 'protein2', 'combined_score']].to_csv(output_csv, index=False)

        print(f"\n✅ 数据已保存: {output_csv}")
        print(f"   • 相互作用数: {len(df_filtered):,}")
        print(f"   • 蛋白质数: {len(set(df_filtered['protein1']) | set(df_filtered['protein2'])):,}")

        # 显示统计信息
        print(f"\n📊 数据统计:")
        print(f"   • 平均置信度: {df_filtered['combined_score'].mean():.1f}")
        print(f"   • 最小置信度: {df_filtered['combined_score'].min()}")
        print(f"   • 最大置信度: {df_filtered['combined_score'].max()}")

        # 删除临时文件
        os.remove(output_file)
        os.remove(uncompressed_file)

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 备选方案: 手动下载")
        print(f"   1. 访问: {string_url}")
        print(f"   2. 保存到: {output_file}")
        print(f"   3. 重新运行此脚本")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_protein_info():
    """下载蛋白质信息 (基因名称映射)"""
    print("\n🧬 下载蛋白质信息 (基因名称映射)")
    print("=" * 70)

    info_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
    output_file = os.path.join(PPI_RAW_DIR, "9606.protein.info.v12.0.txt.gz")

    try:
        response = requests.get(info_url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✅ 下载完成: {output_file}")

        # 解压
        import gzip
        uncompressed_file = output_file.replace('.gz', '')

        with gzip.open(output_file, 'rb') as f_in:
            with open(uncompressed_file, 'wb') as f_out:
                f_out.write(f_in.read())

        # 读取蛋白质信息
        df_info = pd.read_csv(uncompressed_file, sep='\t')

        # 保存简化版本
        df_info['string_protein_id'] = df_info['string_protein_id'].str.replace('9606.', '', regex=False)
        output_csv = os.path.join(PPI_RAW_DIR, "protein_info.csv")
        df_info[['string_protein_id', 'preferred_name', 'protein_size', 'annotation']].to_csv(
            output_csv, index=False
        )

        print(f"✅ 蛋白质信息已保存: {output_csv}")
        print(f"   • 蛋白质数: {len(df_info):,}")

        # 删除临时文件
        os.remove(output_file)
        os.remove(uncompressed_file)

        return True

    except Exception as e:
        print(f"⚠️  蛋白质信息下载失败 (可选): {e}")
        return False


def main():
    print("🔗 步骤1: 下载真实的人类PPI数据")
    print("数据源: STRING数据库 (v12.0)")
    print("物种: 人类 (Homo sapiens, NCBI Taxonomy ID: 9606)")
    print("=" * 70)
    print()

    # 下载PPI数据
    success = download_string_ppi()

    if success:
        # 下载蛋白质信息 (可选)
        download_protein_info()

        print("\n" + "=" * 70)
        print("✅ 步骤1完成: 真实人类PPI数据已下载")
        print(f"📁 数据位置: {PPI_RAW_DIR}")
        print("\n👉 下一步: 下载蛋白质序列数据")
        print("   运行: python 2_download_protein_sequences.py")
        return True
    else:
        print("\n❌ 步骤1失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
