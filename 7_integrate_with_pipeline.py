#!/usr/bin/env python3
"""
步骤7: 集成到主Pipeline
将训练好的边预测器集成到主PPI预测Pipeline中
"""
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

# 导入边预测器模型
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from edge_predictor_augmentation import ImprovedEdgePredictor

# 导入配置
current_dir = os.path.dirname(os.path.abspath(__file__))
import importlib.util
config_path = os.path.join(current_dir, "ppi_config.py")
spec = importlib.util.spec_from_file_location("ppi_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

FEATURE_DIM = config.FEATURE_DIM
DEVICE = config.DEVICE
PPI_PROCESSED_DIR = config.PPI_PROCESSED_DIR


class PPIPredictionPipeline:
    """完整的PPI预测Pipeline"""

    def __init__(self, model_path=None):
        """
        初始化Pipeline

        参数:
            model_path: 训练好的模型路径，如果为None则使用默认路径
        """
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 默认模型路径
        if model_path is None:
            model_path = os.path.join(current_dir, "models", "edge_predictor_best.pth")

        # 加载模型（hidden_dim需要与训练时一致）
        hidden_dim = getattr(config, 'HIDDEN_DIM', 1024)
        self.model = ImprovedEdgePredictor(input_dim=FEATURE_DIM, hidden_dim=hidden_dim).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ 模型已加载: {model_path}")
            print(f"   训练时最佳AUC: {checkpoint.get('best_auc', 0.0):.4f}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载蛋白质特征和映射
        self.features = None
        self.protein_to_idx = None
        self.idx_to_protein = None
        self._load_features()

    def _load_features(self):
        """加载ESM2特征和蛋白质映射"""
        print("\n📊 加载蛋白质特征...")

        features_file = os.path.join(PPI_PROCESSED_DIR, "features.npy")
        mapping_file = os.path.join(PPI_PROCESSED_DIR, "protein_mapping.json")

        if not os.path.exists(features_file):
            raise FileNotFoundError(f"特征文件不存在: {features_file}")

        # 加载特征
        self.features = np.load(features_file)
        print(f"✅ 特征已加载: {self.features.shape}")

        # 加载映射
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
            self.protein_to_idx = mapping['protein_to_idx']
            self.idx_to_protein = {int(k): v for k, v in mapping['idx_to_protein'].items()}

        print(f"✅ 蛋白质映射已加载: {len(self.protein_to_idx)} 个蛋白质")

    def predict_interaction(self, protein1, protein2):
        """
        预测两个蛋白质之间是否存在相互作用

        参数:
            protein1: 蛋白质1的ID (STRING ID)
            protein2: 蛋白质2的ID (STRING ID)

        返回:
            score: 相互作用概率 (0-1)
        """
        # 检查蛋白质是否在数据库中
        if protein1 not in self.protein_to_idx:
            raise ValueError(f"蛋白质 {protein1} 不在数据库中")
        if protein2 not in self.protein_to_idx:
            raise ValueError(f"蛋白质 {protein2} 不在数据库中")

        # 获取特征索引
        idx1 = self.protein_to_idx[protein1]
        idx2 = self.protein_to_idx[protein2]

        # 提取特征
        feat1 = torch.tensor(self.features[idx1], dtype=torch.float32).unsqueeze(0).to(self.device)
        feat2 = torch.tensor(self.features[idx2], dtype=torch.float32).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            score = self.model(feat1, feat2).item()

        return score

    def predict_interactions_batch(self, protein_pairs, batch_size=512):
        """
        批量预测多个蛋白质对之间的相互作用

        参数:
            protein_pairs: 蛋白质对列表 [(protein1, protein2), ...]
            batch_size: 批处理大小

        返回:
            scores: 相互作用概率数组
        """
        print(f"\n🔮 批量预测 {len(protein_pairs):,} 个蛋白质对...")

        all_scores = []

        for i in tqdm(range(0, len(protein_pairs), batch_size), desc="预测"):
            batch_pairs = protein_pairs[i:i+batch_size]

            # 准备批次特征
            src_feats = []
            dst_feats = []

            for p1, p2 in batch_pairs:
                if p1 not in self.protein_to_idx or p2 not in self.protein_to_idx:
                    # 跳过不在数据库中的蛋白质对
                    all_scores.append(0.0)
                    continue

                idx1 = self.protein_to_idx[p1]
                idx2 = self.protein_to_idx[p2]

                src_feats.append(self.features[idx1])
                dst_feats.append(self.features[idx2])

            if len(src_feats) == 0:
                continue

            # 转换为张量
            src_feats = torch.tensor(np.array(src_feats), dtype=torch.float32).to(self.device)
            dst_feats = torch.tensor(np.array(dst_feats), dtype=torch.float32).to(self.device)

            # 预测
            with torch.no_grad():
                batch_scores = self.model(src_feats, dst_feats).squeeze().cpu().numpy()  # [batch, 1] -> [batch]

            all_scores.extend(batch_scores)

        return np.array(all_scores)

    def predict_for_protein(self, protein_id, top_k=100, threshold=0.5):
        """
        预测给定蛋白质与所有其他蛋白质的相互作用

        参数:
            protein_id: 目标蛋白质ID
            top_k: 返回前k个最可能的相互作用
            threshold: 分数阈值，只返回分数高于此值的相互作用

        返回:
            predictions: [(partner_protein, score), ...] 按分数降序排列
        """
        print(f"\n🔍 预测蛋白质 {protein_id} 的潜在相互作用伙伴...")

        if protein_id not in self.protein_to_idx:
            raise ValueError(f"蛋白质 {protein_id} 不在数据库中")

        idx = self.protein_to_idx[protein_id]
        feat = self.features[idx]

        # 与所有其他蛋白质进行预测
        all_scores = []
        all_proteins = []

        batch_size = 512
        num_proteins = len(self.idx_to_protein)

        for i in tqdm(range(0, num_proteins, batch_size), desc="预测"):
            batch_indices = list(range(i, min(i + batch_size, num_proteins)))

            # 跳过自己
            batch_indices = [idx2 for idx2 in batch_indices if idx2 != idx]

            if len(batch_indices) == 0:
                continue

            # 准备批次
            src_feats = np.tile(feat, (len(batch_indices), 1))
            dst_feats = self.features[batch_indices]

            # 转换为张量
            src_feats = torch.tensor(src_feats, dtype=torch.float32).to(self.device)
            dst_feats = torch.tensor(dst_feats, dtype=torch.float32).to(self.device)

            # 预测
            with torch.no_grad():
                batch_scores = self.model(src_feats, dst_feats).squeeze().cpu().numpy()  # [batch, 1] -> [batch]

            all_scores.extend(batch_scores)
            all_proteins.extend([self.idx_to_protein[idx2] for idx2 in batch_indices])

        # 过滤和排序
        predictions = [(p, s) for p, s in zip(all_proteins, all_scores) if s >= threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)

        print(f"✅ 找到 {len(predictions):,} 个潜在相互作用 (分数 >= {threshold})")

        return predictions[:top_k]

    def export_model_for_production(self, output_path=None):
        """
        导出模型用于生产环境

        参数:
            output_path: 导出路径，如果为None则使用默认路径
        """
        if output_path is None:
            output_path = os.path.join(current_dir, "models", "edge_predictor_production.pth")

        print(f"\n📦 导出生产模型...")

        # 保存模型和元数据
        export_dict = {
            'model_state_dict': self.model.state_dict(),
            'feature_dim': FEATURE_DIM,
            'num_proteins': len(self.protein_to_idx),
            'device': str(self.device)
        }

        torch.save(export_dict, output_path)

        print(f"✅ 生产模型已导出: {output_path}")

        # 创建使用说明
        readme_path = os.path.join(os.path.dirname(output_path), "USAGE.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# PPI预测模型使用说明\n\n")
            f.write("## 快速开始\n\n")
            f.write("```python\n")
            f.write("from integrate_with_pipeline import PPIPredictionPipeline\n\n")
            f.write("# 初始化Pipeline\n")
            f.write("pipeline = PPIPredictionPipeline()\n\n")
            f.write("# 预测单个蛋白质对\n")
            f.write('score = pipeline.predict_interaction("ENSP00000000001", "ENSP00000000002")\n')
            f.write('print(f"相互作用概率: {score:.4f}")\n\n')
            f.write("# 批量预测\n")
            f.write('pairs = [("ENSP00000000001", "ENSP00000000002"), ...]\n')
            f.write("scores = pipeline.predict_interactions_batch(pairs)\n\n")
            f.write("# 预测某个蛋白质的所有潜在伙伴\n")
            f.write('partners = pipeline.predict_for_protein("ENSP00000000001", top_k=50)\n')
            f.write("```\n\n")
            f.write("## 模型信息\n\n")
            f.write(f"- 特征维度: {FEATURE_DIM}\n")
            f.write(f"- 蛋白质数量: {len(self.protein_to_idx):,}\n")
            f.write(f"- 特征提取: ESM2 (facebook/esm2_t33_650M_UR50D)\n")
            f.write(f"- 数据来源: STRING v12.0 (Homo sapiens)\n")

        print(f"✅ 使用说明已创建: {readme_path}")


def demo():
    """示例：如何使用Pipeline"""
    print("\n" + "=" * 70)
    print("🎯 Pipeline使用示例")
    print("=" * 70)

    # 初始化Pipeline
    pipeline = PPIPredictionPipeline()

    # 加载测试数据以获取一些示例蛋白质对
    edges_test = np.load(os.path.join(PPI_PROCESSED_DIR, "edges_test.npy"))
    labels_test = np.load(os.path.join(PPI_PROCESSED_DIR, "labels_test.npy"))

    # 获取一些正样本和负样本
    pos_indices = np.where(labels_test == 1)[0][:5]
    neg_indices = np.where(labels_test == 0)[0][:5]

    print("\n📊 示例1: 预测已知的正样本（真实相互作用）")
    print("-" * 70)
    for idx in pos_indices:
        src_idx, dst_idx = edges_test[idx]
        p1 = pipeline.idx_to_protein[src_idx]
        p2 = pipeline.idx_to_protein[dst_idx]

        score = pipeline.predict_interaction(p1, p2)
        print(f"  {p1} <-> {p2}: {score:.4f} {'✅' if score > 0.5 else '❌'}")

    print("\n📊 示例2: 预测负样本（无相互作用）")
    print("-" * 70)
    for idx in neg_indices:
        src_idx, dst_idx = edges_test[idx]
        p1 = pipeline.idx_to_protein[src_idx]
        p2 = pipeline.idx_to_protein[dst_idx]

        score = pipeline.predict_interaction(p1, p2)
        print(f"  {p1} <-> {p2}: {score:.4f} {'❌' if score < 0.5 else '⚠️'}")

    # 示例3: 批量预测
    print("\n📊 示例3: 批量预测")
    print("-" * 70)
    sample_pairs = []
    for idx in pos_indices[:3]:
        src_idx, dst_idx = edges_test[idx]
        p1 = pipeline.idx_to_protein[src_idx]
        p2 = pipeline.idx_to_protein[dst_idx]
        sample_pairs.append((p1, p2))

    scores = pipeline.predict_interactions_batch(sample_pairs)
    for (p1, p2), score in zip(sample_pairs, scores):
        print(f"  {p1} <-> {p2}: {score:.4f}")

    # 示例4: 预测某个蛋白质的潜在伙伴
    print("\n📊 示例4: 预测蛋白质的Top-10潜在相互作用伙伴")
    print("-" * 70)
    sample_protein = pipeline.idx_to_protein[edges_test[pos_indices[0]][0]]
    print(f"  目标蛋白质: {sample_protein}")

    partners = pipeline.predict_for_protein(sample_protein, top_k=10, threshold=0.5)
    for i, (partner, score) in enumerate(partners, 1):
        print(f"  {i:2d}. {partner}: {score:.4f}")

    # 导出生产模型
    pipeline.export_model_for_production()


def main():
    print("🔗 步骤7: 集成到主Pipeline")
    print("=" * 70)

    # 检查模型是否存在
    model_path = os.path.join(current_dir, "models", "edge_predictor_best.pth")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("请先运行: python 5_train_edge_predictor.py")
        return False

    # 运行示例
    try:
        demo()
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ 步骤7完成: Pipeline集成完成")
    print("\n📚 使用方法:")
    print("```python")
    print("from integrate_with_pipeline import PPIPredictionPipeline")
    print("")
    print("# 初始化")
    print("pipeline = PPIPredictionPipeline()")
    print("")
    print("# 预测单个蛋白质对")
    print('score = pipeline.predict_interaction("PROTEIN1", "PROTEIN2")')
    print("")
    print("# 批量预测")
    print("scores = pipeline.predict_interactions_batch(protein_pairs)")
    print("")
    print("# 预测某个蛋白质的潜在伙伴")
    print('partners = pipeline.predict_for_protein("PROTEIN_ID", top_k=100)')
    print("```")

    print("\n🎉 完整训练流程已完成!")
    print("\n📁 输出文件:")
    print("   • 训练好的模型: models/edge_predictor_best.pth")
    print("   • 生产模型: models/edge_predictor_production.pth")
    print("   • 评估结果: results/")
    print("   • 使用说明: models/USAGE.md")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
