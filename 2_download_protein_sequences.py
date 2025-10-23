#!/usr/bin/env python3
"""
下载蛋白质序列数据
从 UniProt 下载STRING蛋白质ID对应的氨基酸序列
"""
import os
import sys
import pandas as pd
import requests
from tqdm import tqdm
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")


def download_string_sequences():
    """
    从STRING数据库下载蛋白质序列
    STRING提供了映射到UniProt的序列数据
    """
    print("🧬 下载蛋白质序列数据")
    print("=" * 70)

    # STRING序列文件
    sequences_url = "https://stringdb-downloads.org/download/protein.sequences.v12.0/9606.protein.sequences.v12.0.fa.gz"
    output_file = os.path.join(PPI_RAW_DIR, "9606.protein.sequences.v12.0.fa.gz")

    print(f"📥 下载URL: {sequences_url}")
    print(f"💾 保存位置: {output_file}")

    try:
        response = requests.get(sequences_url, stream=True, timeout=600)
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
        print("\n📦 解压序列文件...")
        import gzip

        uncompressed_file = output_file.replace('.gz', '')

        with gzip.open(output_file, 'rb') as f_in:
            with open(uncompressed_file, 'wb') as f_out:
                f_out.write(f_in.read())

        print(f"✅ 解压完成: {uncompressed_file}")

        # 解析FASTA文件
        print("\n📊 解析FASTA序列...")
        sequences = {}
        current_id = None
        current_seq = []

        with open(uncompressed_file, 'r') as f:
            for line in tqdm(f, desc="解析序列"):
                line = line.strip()
                if line.startswith('>'):
                    # 保存前一个序列
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)

                    # 提取蛋白质ID (去掉 "9606." 前缀)
                    current_id = line[1:].split()[0].replace('9606.', '')
                    current_seq = []
                else:
                    current_seq.append(line)

            # 保存最后一个序列
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)

        print(f"✅ 解析完成: {len(sequences):,} 条序列")

        # 加载PPI数据以获取需要的蛋白质列表
        ppi_file = os.path.join(PPI_RAW_DIR, "ppi_data.csv")
        df_ppi = pd.read_csv(ppi_file)

        proteins_needed = set(df_ppi['protein1']) | set(df_ppi['protein2'])
        print(f"\n📊 PPI数据中的蛋白质数: {len(proteins_needed):,}")

        # 筛选需要的序列
        filtered_sequences = {k: v for k, v in sequences.items() if k in proteins_needed}

        print(f"✅ 匹配的序列数: {len(filtered_sequences):,}")
        print(f"⚠️  缺失的蛋白质数: {len(proteins_needed - set(filtered_sequences.keys())):,}")

        # 保存序列为FASTA格式
        output_fasta = os.path.join(PPI_RAW_DIR, "protein_sequences.fasta")
        with open(output_fasta, 'w') as f:
            for protein_id, sequence in tqdm(filtered_sequences.items(), desc="保存序列"):
                f.write(f">{protein_id}\n")
                # 每行60个字符
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + '\n')

        print(f"\n✅ 序列已保存: {output_fasta}")

        # 统计信息
        seq_lengths = [len(seq) for seq in filtered_sequences.values()]
        print(f"\n📊 序列统计:")
        print(f"   • 最短序列: {min(seq_lengths)} 氨基酸")
        print(f"   • 最长序列: {max(seq_lengths)} 氨基酸")
        print(f"   • 平均长度: {sum(seq_lengths) / len(seq_lengths):.0f} 氨基酸")

        # 删除临时文件
        os.remove(output_file)
        os.remove(uncompressed_file)

        # 保存蛋白质ID列表
        protein_list_file = os.path.join(PPI_RAW_DIR, "protein_list.txt")
        with open(protein_list_file, 'w') as f:
            for protein_id in sorted(filtered_sequences.keys()):
                f.write(f"{protein_id}\n")

        print(f"✅ 蛋白质列表已保存: {protein_list_file}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🧬 步骤2: 下载蛋白质序列数据")
    print("数据源: STRING数据库 (v12.0)")
    print("=" * 70)
    print()

    success = download_string_sequences()

    if success:
        print("\n" + "=" * 70)
        print("✅ 步骤2完成: 蛋白质序列数据已下载")
        print(f"📁 数据位置: {PPI_RAW_DIR}")
        print("\n👉 下一步: 使用ESM2提取特征")
        print("   运行: python 3_extract_esm2_features.py")
        return True
    else:
        print("\n❌ 步骤2失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
