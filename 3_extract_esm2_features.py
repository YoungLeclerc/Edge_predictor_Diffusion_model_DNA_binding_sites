#!/usr/bin/env python3
"""
使用ESM2提取蛋白质特征
ESM2 (Evolutionary Scale Modeling 2) 是Meta AI开发的蛋白质语言模型
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")
PPI_PROCESSED_DIR = os.path.join(DATA_DIR, "ppi_processed")
os.makedirs(PPI_PROCESSED_DIR, exist_ok=True)


def load_sequences(fasta_file):
    """从FASTA文件加载序列"""
    print(f"📖 加载FASTA文件: {fasta_file}")

    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)

                # 提取蛋白质ID
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # 保存最后一个序列
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    print(f"✅ 加载完成: {len(sequences):,} 条序列")
    return sequences


def extract_esm2_features(sequences, model_name="facebook/esm2_t33_650M_UR50D", batch_size=8):
    """
    使用ESM2提取蛋白质特征

    参数:
        sequences: dict, {protein_id: sequence}
        model_name: ESM2模型名称
        batch_size: 批处理大小 (根据GPU内存调整)
    """
    print(f"\n🧬 使用ESM2提取特征...")
    print(f"   模型: {model_name}")
    print(f"   批大小: {batch_size}")

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   设备: {device}")

    if device.type == "cpu":
        print("   ⚠️  警告: 使用CPU，速度会很慢！建议使用GPU")

    # 加载模型和分词器
    print("\n📥 加载ESM2模型 (首次加载会下载 ~2.5GB)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n💡 解决方案:")
        print("   1. 检查网络连接")
        print("   2. 安装transformers: pip install transformers")
        print("   3. 手动下载模型后重试")
        return None

    # 准备序列列表 (按蛋白质ID排序以保持一致性)
    protein_ids = sorted(sequences.keys())
    sequences_list = [sequences[pid] for pid in protein_ids]

    print(f"\n🔄 提取 {len(sequences_list):,} 个蛋白质的特征...")
    print(f"   预计时间: ~{len(sequences_list) * 0.5 / batch_size / 60:.1f} 分钟 (GPU)")

    all_features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences_list), batch_size), desc="提取特征"):
            batch_seqs = sequences_list[i:i+batch_size]

            try:
                # 分词 (截断到1024个token)
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 获取模型输出
                outputs = model(**inputs, output_hidden_states=True)

                # 使用最后一层的平均池化作为特征
                last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, 1280]

                # 去掉特殊token (padding) 后进行平均池化
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.shape).float()
                masked_hidden = last_hidden * mask
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = mask.sum(dim=1)
                batch_features = (sum_hidden / lengths).cpu().numpy()

                all_features.append(batch_features)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  GPU内存不足，减小batch_size并重试")
                    print(f"   当前batch_size={batch_size}, 建议减小到 {batch_size // 2}")
                    torch.cuda.empty_cache()
                    return None
                else:
                    raise e

    # 合并所有批次
    features = np.vstack(all_features).astype(np.float32)

    print(f"\n✅ 特征提取完成!")
    print(f"   • 形状: {features.shape}")
    print(f"   • 类型: {features.dtype}")
    print(f"   • 大小: {features.nbytes / 1024 / 1024:.1f} MB")
    print(f"\n📊 特征统计:")
    print(f"   • 均值: {features.mean():.6f}")
    print(f"   • 标准差: {features.std():.6f}")
    print(f"   • 最小值: {features.min():.6f}")
    print(f"   • 最大值: {features.max():.6f}")

    return features, protein_ids


def save_features(features, protein_ids, output_dir):
    """保存特征"""
    print(f"\n💾 保存特征...")

    # 保存特征矩阵
    features_file = os.path.join(output_dir, "features.npy")
    np.save(features_file, features)
    print(f"✅ 特征已保存: {features_file}")

    # 保存蛋白质ID映射
    protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    idx_to_protein = {i: pid for pid, i in protein_to_idx.items()}

    import json
    mapping_file = os.path.join(output_dir, "protein_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump({
            'protein_to_idx': protein_to_idx,
            'idx_to_protein': idx_to_protein,
            'num_proteins': len(protein_ids),
            'feature_dim': features.shape[1]
        }, f, indent=2)

    print(f"✅ 映射已保存: {mapping_file}")


def main():
    print("🧬 步骤3: 使用ESM2提取蛋白质特征")
    print("=" * 70)
    print()

    # 加载序列
    fasta_file = os.path.join(PPI_RAW_DIR, "protein_sequences.fasta")
    if not os.path.exists(fasta_file):
        print(f"❌ 错误: 序列文件不存在: {fasta_file}")
        print("请先运行: python 2_download_protein_sequences.py")
        return False

    sequences = load_sequences(fasta_file)

    # 提取ESM2特征
    # 注意: batch_size根据GPU内存调整
    # - 16GB GPU: batch_size=16
    # - 24GB GPU: batch_size=24
    # - 40GB GPU: batch_size=32
    # - CPU: batch_size=1 (非常慢)

    batch_size = 64  # 适合A100 40GB
    result = extract_esm2_features(sequences, batch_size=batch_size)

    if result is None:
        print("\n❌ 特征提取失败")
        return False

    features, protein_ids = result

    # 保存特征
    save_features(features, protein_ids, PPI_PROCESSED_DIR)

    print("\n" + "=" * 70)
    print("✅ 步骤3完成: ESM2特征已提取")
    print(f"📁 特征位置: {PPI_PROCESSED_DIR}")
    print(f"   • 特征文件: features.npy ({features.shape})")
    print(f"   • 映射文件: protein_mapping.json")
    print("\n👉 下一步: 预处理PPI数据")
    print("   运行: python 4_preprocess_ppi_data.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
