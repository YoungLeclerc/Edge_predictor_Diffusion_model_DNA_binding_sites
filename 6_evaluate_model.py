#!/usr/bin/env python3
"""
步骤6: 评估边预测器模型
在测试集上评估训练好的模型，生成详细的评估报告
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from torch.utils.data import TensorDataset, DataLoader
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
BATCH_SIZE = config.BATCH_SIZE
PPI_PROCESSED_DIR = config.PPI_PROCESSED_DIR


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 加载模型（hidden_dim需要与训练时一致）
        hidden_dim = getattr(config, 'HIDDEN_DIM', 1024)
        self.model = ImprovedEdgePredictor(input_dim=FEATURE_DIM, hidden_dim=hidden_dim).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.best_auc = checkpoint.get('best_auc', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        print(f"✅ 模型已加载: {model_path}")
        print(f"   • 训练时最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")

    def load_test_data(self):
        """加载测试数据"""
        print("\n📊 加载测试数据...")

        # 加载测试边和标签
        edges_test = np.load(os.path.join(PPI_PROCESSED_DIR, "edges_test.npy"))
        labels_test = np.load(os.path.join(PPI_PROCESSED_DIR, "labels_test.npy"))

        # 加载特征
        features = np.load(os.path.join(PPI_PROCESSED_DIR, "features.npy"))

        print(f"✅ 测试数据加载完成:")
        print(f"   • 测试边: {len(edges_test):,}")
        print(f"     └─ 正样本: {(labels_test==1).sum():,}, 负样本: {(labels_test==0).sum():,}")
        print(f"   • 特征维度: {features.shape}")

        return edges_test, labels_test, features

    def create_dataloader(self, edges, labels, features):
        """创建数据加载器"""
        src_feats = torch.tensor(features[edges[:, 0]], dtype=torch.float32)
        dst_feats = torch.tensor(features[edges[:, 1]], dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(src_feats, dst_feats, labels_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        return dataloader

    def predict(self, dataloader):
        """在数据集上进行预测"""
        print("\n🔮 进行预测...")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for src_feat, dst_feat, labels in tqdm(dataloader, desc="预测"):
                src_feat = src_feat.to(self.device)
                dst_feat = dst_feat.to(self.device)

                predictions = self.model(src_feat, dst_feat).squeeze()  # [batch, 1] -> [batch]

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return all_preds, all_labels

    def calculate_metrics(self, preds, labels, threshold=0.5):
        """计算评估指标"""
        print("\n📈 计算评估指标...")

        # 二分类预测
        binary_preds = (preds > threshold).astype(int)

        # 计算指标
        metrics = {
            'auc': roc_auc_score(labels, preds),
            'accuracy': accuracy_score(labels, binary_preds),
            'precision': precision_score(labels, binary_preds, zero_division=0),
            'recall': recall_score(labels, binary_preds, zero_division=0),
            'f1': f1_score(labels, binary_preds, zero_division=0),
            'average_precision': average_precision_score(labels, preds),
            'threshold': threshold
        }

        # 混淆矩阵
        cm = confusion_matrix(labels, binary_preds)
        tn, fp, fn, tp = cm.ravel()

        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }

        # 特异性和敏感性
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def plot_roc_curve(self, preds, labels, save_path):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(labels, preds)
        auc = roc_auc_score(labels, preds)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ ROC曲线已保存: {save_path}")

    def plot_precision_recall_curve(self, preds, labels, save_path):
        """绘制Precision-Recall曲线"""
        precision, recall, thresholds = precision_recall_curve(labels, preds)
        ap = average_precision_score(labels, preds)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ PR曲线已保存: {save_path}")

    def plot_confusion_matrix(self, cm_dict, save_path):
        """绘制混淆矩阵"""
        cm = np.array([
            [cm_dict['true_negative'], cm_dict['false_positive']],
            [cm_dict['false_negative'], cm_dict['true_positive']]
        ])

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()

        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)

        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)

        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 混淆矩阵已保存: {save_path}")

    def save_evaluation_report(self, metrics, save_path):
        """保存评估报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("边预测器模型评估报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"训练时最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})\n\n")

            f.write("测试集性能:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  AUC-ROC:           {metrics['auc']:.4f}\n")
            f.write(f"  Average Precision: {metrics['average_precision']:.4f}\n")
            f.write(f"  Accuracy:          {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision:         {metrics['precision']:.4f}\n")
            f.write(f"  Recall:            {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:          {metrics['f1']:.4f}\n")
            f.write(f"  Sensitivity:       {metrics['sensitivity']:.4f}\n")
            f.write(f"  Specificity:       {metrics['specificity']:.4f}\n")
            f.write(f"  Decision Threshold: {metrics['threshold']:.2f}\n\n")

            f.write("混淆矩阵:\n")
            f.write("-" * 70 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"  True Negative:  {cm['true_negative']:,}\n")
            f.write(f"  False Positive: {cm['false_positive']:,}\n")
            f.write(f"  False Negative: {cm['false_negative']:,}\n")
            f.write(f"  True Positive:  {cm['true_positive']:,}\n\n")

            f.write("性能评估:\n")
            f.write("-" * 70 + "\n")
            if metrics['auc'] > 0.65:
                f.write("  ✅ AUC显著高于随机基线(0.50)，模型学习到了有效的PPI模式\n")
            else:
                f.write("  ⚠️  AUC较低，可能需要:\n")
                f.write("     - 调整超参数\n")
                f.write("     - 增加训练数据\n")
                f.write("     - 使用更复杂的模型架构\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"✅ 评估报告已保存: {save_path}")


def main():
    print("📊 步骤6: 评估边预测器模型 (超稳定版)")
    print("=" * 70)

    # 模型路径 - 使用超稳定版
    model_path = os.path.join(current_dir, "models", "edge_predictor_best_ultra_stable.pth")

    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("请先运行: python 5_train_edge_predictor_ultra_stable.py")
        return False

    print(f"📁 评估模型: edge_predictor_best_ultra_stable.pth")

    # 创建评估器
    evaluator = ModelEvaluator(model_path)

    # 加载测试数据
    edges_test, labels_test, features = evaluator.load_test_data()

    # 创建数据加载器
    test_loader = evaluator.create_dataloader(edges_test, labels_test, features)

    # 进行预测
    preds, labels = evaluator.predict(test_loader)

    # 计算指标
    metrics = evaluator.calculate_metrics(preds, labels)

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 测试集评估结果")
    print("=" * 70)
    print(f"  AUC-ROC:           {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Sensitivity:       {metrics['sensitivity']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")

    # 创建结果目录 - 超稳定版专用目录
    results_dir = os.path.join(current_dir, "results_ultra_stable")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n💾 结果将保存到: {results_dir}")

    # 绘制可视化
    print("\n📈 生成可视化...")
    evaluator.plot_roc_curve(
        preds, labels,
        os.path.join(results_dir, "roc_curve_ultra_stable.png")
    )
    evaluator.plot_precision_recall_curve(
        preds, labels,
        os.path.join(results_dir, "precision_recall_curve_ultra_stable.png")
    )
    evaluator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(results_dir, "confusion_matrix_ultra_stable.png")
    )

    # 保存指标
    metrics_file = os.path.join(results_dir, "test_metrics_ultra_stable.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ 指标已保存: {metrics_file}")

    # 保存评估报告
    report_file = os.path.join(results_dir, "evaluation_report_ultra_stable.txt")
    evaluator.save_evaluation_report(metrics, report_file)

    print("\n" + "=" * 70)
    print("✅ 步骤6完成: 超稳定版模型评估完成")
    print(f"📁 结果位置: {results_dir}/")
    print(f"   • ROC曲线: roc_curve_ultra_stable.png")
    print(f"   • PR曲线: precision_recall_curve_ultra_stable.png")
    print(f"   • 混淆矩阵: confusion_matrix_ultra_stable.png")
    print(f"   • 评估报告: evaluation_report_ultra_stable.txt")

    if metrics['auc'] > 0.65:
        print("\n🎉 模型性能优秀! AUC显著高于随机基线")
    else:
        print("\n⚠️  模型性能较低，建议检查训练过程或调整超参数")

    print("\n👉 下一步: 集成到Pipeline")
    print("   运行: python robust_pipeline_edge.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
