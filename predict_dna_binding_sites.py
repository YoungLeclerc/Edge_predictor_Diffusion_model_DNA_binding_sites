#!/usr/bin/env python3
"""
完整的DNA结合位点预测和可视化脚本

功能:
1. 从蛋白质序列提取ESM-2特征
2. 使用训练好的GNN模型预测DNA结合位点
3. 生成PyMOL可视化脚本
4. 支持DNA-573和DNA-646两个训练模型

作者: Advanced GAT-GNN DNA Binding Site Predictor
日期: 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入模型
from advanced_gnn_model import AdvancedBindingSiteGNN


class DNABindingSitePredictor:
    """DNA结合位点预测器"""

    def __init__(self, model_path, device='cuda'):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
            device: 计算设备 ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")

        # 加载模型
        print(f"📥 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # 获取模型配置
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            self.model = AdvancedBindingSiteGNN(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型加载成功 (使用保存的配置)")
        else:
            # 使用默认配置
            self.model = AdvancedBindingSiteGNN(
                input_dim=1280,  # ESM-2特征维度
                hidden_dim=256,
                num_layers=4,
                heads=4,
                dropout=0.3,
                use_edge_features=True
            )
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ 模型加载成功 (使用默认配置)")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载ESM-2模型
        print(f"📥 加载ESM-2模型...")
        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = "facebook/esm2_t33_650M_UR50D"
            # 使用local_files_only=True优先使用本地缓存
            self.esm_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.esm_model = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            print(f"✅ ESM-2模型加载成功 (使用本地缓存)")
        except Exception as e:
            print(f"❌ ESM-2模型加载失败: {e}")
            print("请安装transformers: pip install transformers")
            print("或首次运行需要下载ESM-2模型（~2.5GB）")
            raise

    def extract_esm2_features(self, sequence):
        """
        从蛋白质序列提取ESM-2特征

        Args:
            sequence: 蛋白质序列字符串

        Returns:
            features: (seq_len, 1280) 每个残基的特征向量
        """
        print(f"🧬 提取ESM-2特征 (序列长度: {len(sequence)})")

        with torch.no_grad():
            # 分词
            inputs = self.esm_tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取模型输出
            outputs = self.esm_model(**inputs, output_hidden_states=True)

            # 使用最后一层的隐藏状态
            # Shape: (1, seq_len+2, 1280) - +2是因为有<cls>和<eos> token
            last_hidden = outputs.last_hidden_state

            # 去掉特殊token，只保留序列部分
            # [0, 1:-1] -> 去掉batch维度，去掉<cls>和<eos>
            sequence_features = last_hidden[0, 1:-1, :].cpu().numpy()

        print(f"✅ 特征提取完成: {sequence_features.shape}")
        return sequence_features

    def build_graph_from_sequence(self, sequence_features, k_neighbors=10):
        """
        从序列特征构建图

        Args:
            sequence_features: (seq_len, 1280) 特征矩阵
            k_neighbors: KNN构建边的邻居数

        Returns:
            data: PyG Data对象
        """
        seq_len = sequence_features.shape[0]

        # 节点特征
        x = torch.tensor(sequence_features, dtype=torch.float32)

        # 构建边: 使用序列位置 + KNN
        edge_index = []

        # 1. 序列邻接边 (i, i+1)
        for i in range(seq_len - 1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])

        # 2. KNN边 (基于特征相似度)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, seq_len), algorithm='auto')
        nbrs.fit(sequence_features)
        distances, indices = nbrs.kneighbors(sequence_features)

        for i in range(seq_len):
            for j in indices[i][1:]:  # 跳过自己
                if i != j:
                    edge_index.append([i, j])

        # 转换为tensor
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 创建Data对象并添加batch信息（单个图，所有节点batch=0）
        batch = torch.zeros(seq_len, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)

        print(f"📊 图构建完成: {seq_len} 节点, {edge_index.shape[1]} 条边")
        return data

    def predict(self, sequence, threshold=0.5, k_neighbors=10):
        """
        预测DNA结合位点

        Args:
            sequence: 蛋白质序列
            threshold: 分类阈值
            k_neighbors: 图构建的邻居数

        Returns:
            predictions: (seq_len,) 预测概率
            binding_sites: (seq_len,) 二值标签 (0/1)
        """
        # 1. 提取ESM-2特征
        features = self.extract_esm2_features(sequence)

        # 2. 构建图
        data = self.build_graph_from_sequence(features, k_neighbors=k_neighbors)
        data = data.to(self.device)

        # 3. 模型预测
        print(f"🔮 开始预测...")
        with torch.no_grad():
            logits = self.model(data)
            probs = torch.sigmoid(logits).cpu().numpy()

        # 4. 阈值化
        binding_sites = (probs >= threshold).astype(int)

        num_binding = binding_sites.sum()
        ratio = num_binding / len(sequence) * 100

        print(f"✅ 预测完成!")
        print(f"   • 序列长度: {len(sequence)}")
        print(f"   • 结合位点: {num_binding} ({ratio:.1f}%)")
        print(f"   • 非结合位点: {len(sequence) - num_binding} ({100-ratio:.1f}%)")

        return probs, binding_sites


def generate_pymol_script(sequence, predictions, binding_sites,
                         output_file, pdb_id=None, pdb_file=None,
                         high_conf_threshold=0.7, low_conf_threshold=0.5):
    """
    生成PyMOL可视化脚本

    Args:
        sequence: 蛋白质序列
        predictions: 预测概率
        binding_sites: 二值标签
        output_file: 输出.pml文件路径
        pdb_id: PDB ID (如果有)
        pdb_file: 本地PDB文件路径
        high_conf_threshold: 高置信度阈值
        low_conf_threshold: 低置信度阈值
    """
    script = []

    # 加载结构
    script.append("# DNA Binding Site Visualization")
    script.append("# Generated by Advanced GAT-GNN Predictor\n")

    if pdb_file:
        script.append(f"load {pdb_file}, protein")
    elif pdb_id:
        script.append(f"fetch {pdb_id}, protein")
    else:
        script.append("# No structure provided - please load manually")
        script.append("# load your_structure.pdb, protein\n")

    # 基础设置
    script.append("\n# Basic settings")
    script.append("bg_color white")
    script.append("hide everything")
    script.append("show cartoon, protein")
    script.append("color gray80, protein")
    script.append("set cartoon_fancy_helices, 1")
    script.append("set cartoon_smooth_loops, 1\n")

    # 按置信度分类残基
    high_conf_residues = []
    medium_conf_residues = []
    low_conf_residues = []

    for i, (pred, is_binding) in enumerate(zip(predictions, binding_sites)):
        if is_binding:
            residue_num = i + 1  # 残基编号从1开始
            if pred >= high_conf_threshold:
                high_conf_residues.append(residue_num)
            elif pred >= low_conf_threshold:
                medium_conf_residues.append(residue_num)
            else:
                low_conf_residues.append(residue_num)

    # 创建选择和着色
    script.append("# DNA binding sites\n")

    if high_conf_residues:
        residues_str = "+".join(map(str, high_conf_residues))
        script.append(f"# High confidence binding sites ({len(high_conf_residues)} residues)")
        script.append(f"select high_conf, resi {residues_str}")
        script.append("color red, high_conf")
        script.append("show sticks, high_conf")
        script.append("show spheres, high_conf")
        script.append("set sphere_scale, 0.3, high_conf\n")

    if medium_conf_residues:
        residues_str = "+".join(map(str, medium_conf_residues))
        script.append(f"# Medium confidence binding sites ({len(medium_conf_residues)} residues)")
        script.append(f"select medium_conf, resi {residues_str}")
        script.append("color orange, medium_conf")
        script.append("show sticks, medium_conf")
        script.append("show spheres, medium_conf")
        script.append("set sphere_scale, 0.25, medium_conf\n")

    if low_conf_residues:
        residues_str = "+".join(map(str, low_conf_residues))
        script.append(f"# Low confidence binding sites ({len(low_conf_residues)} residues)")
        script.append(f"select low_conf, resi {residues_str}")
        script.append("color yellow, low_conf")
        script.append("show sticks, low_conf\n")

    # 视图设置
    script.append("# View settings")
    script.append("orient")
    script.append("zoom protein")
    script.append("set ray_shadows, 0")
    script.append("set antialias, 2")
    script.append("set orthoscopic, on\n")

    # 标签
    script.append("# Labels")
    script.append("set label_size, 20")
    script.append("set label_color, black\n")

    # 图例
    script.append("# Legend")
    script.append("# Red: High confidence (p >= {:.2f})".format(high_conf_threshold))
    script.append("# Orange: Medium confidence ({:.2f} <= p < {:.2f})".format(low_conf_threshold, high_conf_threshold))
    script.append("# Yellow: Low confidence (p < {:.2f})".format(low_conf_threshold))
    script.append("# Gray: Non-binding sites\n")

    # 保存脚本
    with open(output_file, 'w') as f:
        f.write('\n'.join(script))

    print(f"✅ PyMOL脚本已保存: {output_file}")


def save_results(sequence, predictions, binding_sites, output_dir, seq_id=None):
    """
    保存预测结果

    Args:
        sequence: 蛋白质序列
        predictions: 预测概率
        binding_sites: 二值标签
        output_dir: 输出目录
        seq_id: 序列ID
    """
    os.makedirs(output_dir, exist_ok=True)

    seq_id = seq_id or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. JSON格式 (详细)
    json_file = os.path.join(output_dir, f"{seq_id}_predictions.json")
    results = {
        'sequence_id': seq_id,
        'sequence': sequence,
        'sequence_length': len(sequence),
        'predictions': predictions.tolist(),
        'binding_sites': binding_sites.tolist(),
        'num_binding_sites': int(binding_sites.sum()),
        'binding_ratio': float(binding_sites.sum() / len(sequence)),
        'timestamp': timestamp
    }

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON结果已保存: {json_file}")

    # 2. 文本格式 (人类可读)
    txt_file = os.path.join(output_dir, f"{seq_id}_predictions.txt")
    with open(txt_file, 'w') as f:
        f.write(f"DNA Binding Site Predictions\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Sequence ID: {seq_id}\n")
        f.write(f"Sequence Length: {len(sequence)}\n")
        f.write(f"Binding Sites: {binding_sites.sum()} ({binding_sites.sum()/len(sequence)*100:.1f}%)\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write(f"{'Position':<10} {'Residue':<10} {'Prediction':<12} {'Binding':<10}\n")
        f.write(f"{'-'*50}\n")

        for i, (aa, pred, is_binding) in enumerate(zip(sequence, predictions, binding_sites)):
            pos = i + 1
            binding_str = "YES" if is_binding else "no"
            f.write(f"{pos:<10} {aa:<10} {pred:.4f}      {binding_str:<10}\n")

    print(f"✅ 文本结果已保存: {txt_file}")

    # 3. FASTA格式 (带标签)
    fasta_file = os.path.join(output_dir, f"{seq_id}_annotated.fasta")
    with open(fasta_file, 'w') as f:
        f.write(f">{seq_id} | DNA binding sites\n")
        f.write(f"{sequence}\n")
        f.write(f">Binding_sites (1=binding, 0=non-binding)\n")
        f.write(f"{''.join(map(str, binding_sites))}\n")
        f.write(f">Prediction_scores\n")
        for pred in predictions:
            f.write(f"{pred:.3f} ")
        f.write("\n")

    print(f"✅ FASTA结果已保存: {fasta_file}")

    return {
        'json': json_file,
        'txt': txt_file,
        'fasta': fasta_file
    }


def main():
    parser = argparse.ArgumentParser(
        description="预测蛋白质序列的DNA结合位点并生成PyMOL可视化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从FASTA文件预测 (使用DNA-573模型)
  python predict_dna_binding_sites.py --fasta protein.fasta --model dna573

  # 从序列字符串预测 (使用DNA-646模型)
  python predict_dna_binding_sites.py --sequence "MKLAVLV..." --model dna646

  # 指定自定义模型路径
  python predict_dna_binding_sites.py --fasta protein.fasta --model-path /path/to/model.pt

  # 生成PyMOL脚本 (使用PDB ID)
  python predict_dna_binding_sites.py --fasta protein.fasta --pdb-id 1ABC

  # 生成PyMOL脚本 (使用本地PDB文件)
  python predict_dna_binding_sites.py --fasta protein.fasta --pdb-file protein.pdb
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--fasta', type=str, help='FASTA文件路径')
    input_group.add_argument('--sequence', type=str, help='蛋白质序列字符串')

    # 模型选项
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, choices=['dna573', 'dna646'],
                           help='使用预训练模型: dna573 或 dna646')
    model_group.add_argument('--model-path', type=str,
                           help='自定义模型路径')

    # 预测参数
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='分类阈值 (默认: 0.5)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='图构建的KNN邻居数 (默认: 10)')

    # PyMOL可视化
    parser.add_argument('--pdb-id', type=str, help='PDB ID (用于PyMOL可视化)')
    parser.add_argument('--pdb-file', type=str, help='本地PDB文件路径')

    # 输出选项
    parser.add_argument('--output-dir', type=str, default='prediction_results',
                       help='输出目录 (默认: prediction_results)')
    parser.add_argument('--seq-id', type=str, help='序列ID (用于输出文件命名)')

    # 设备选项
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备 (默认: cuda)')

    args = parser.parse_args()

    # 打印标题
    print("=" * 80)
    print("🧬 DNA结合位点预测器 - Advanced GAT-GNN")
    print("=" * 80)
    print()

    # 1. 加载序列
    if args.fasta:
        print(f"📖 从FASTA文件加载序列: {args.fasta}")
        with open(args.fasta, 'r') as f:
            lines = f.readlines()

        sequence = ""
        seq_id = None
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if seq_id is None:  # 只读取第一条序列
                    seq_id = line[1:].split()[0]
            else:
                if all(c in '01' for c in line):  # 跳过标签行
                    continue
                sequence += line

        if not sequence:
            print("❌ 错误: FASTA文件中没有找到有效序列")
            return

        seq_id = args.seq_id or seq_id or "unknown"
        print(f"✅ 序列加载成功")
        print(f"   • ID: {seq_id}")
        print(f"   • 长度: {len(sequence)}")
    else:
        sequence = args.sequence
        seq_id = args.seq_id or "custom_sequence"
        print(f"✅ 使用提供的序列 (长度: {len(sequence)})")

    # 2. 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        base_dir = "/mnt/data2/Yang/zhq_pro/method2_ppi_training/Augmented_data_balanced"
        if args.model == 'dna573':
            model_path = os.path.join(base_dir, "DNA-573_Train_ultimate_r050/ultimate_gnn_model.pt")
        else:  # dna646
            model_path = os.path.join(base_dir, "DNA-646_Train_ultimate_r050/ultimate_gnn_model.pt")

    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return

    print(f"\n📊 使用模型: {model_path}")

    # 3. 创建预测器
    try:
        predictor = DNABindingSitePredictor(model_path, device=args.device)
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        return

    # 4. 预测
    print(f"\n{'='*80}")
    try:
        predictions, binding_sites = predictor.predict(
            sequence,
            threshold=args.threshold,
            k_neighbors=args.k_neighbors
        )
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 保存结果
    print(f"\n{'='*80}")
    print(f"💾 保存结果...")
    output_files = save_results(sequence, predictions, binding_sites, args.output_dir, seq_id)

    # 6. 生成PyMOL脚本
    if args.pdb_id or args.pdb_file:
        print(f"\n📊 生成PyMOL可视化脚本...")
        pml_file = os.path.join(args.output_dir, f"{seq_id}_visualization.pml")
        generate_pymol_script(
            sequence, predictions, binding_sites,
            pml_file,
            pdb_id=args.pdb_id,
            pdb_file=args.pdb_file
        )
        print(f"\n💡 使用PyMOL可视化:")
        print(f"   pymol {pml_file}")

    # 总结
    print(f"\n{'='*80}")
    print(f"✅ 预测完成!")
    print(f"\n📁 输出文件:")
    for key, path in output_files.items():
        print(f"   • {key.upper()}: {path}")
    if args.pdb_id or args.pdb_file:
        print(f"   • PyMOL: {pml_file}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
