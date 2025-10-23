#!/usr/bin/env python3
"""
步骤5: 训练边预测器（真实PPI数据 + 真实ESM2特征）
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
EPOCHS = config.TRAIN_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
PPI_PROCESSED_DIR = config.PPI_PROCESSED_DIR


class EdgePredictorTrainer:
    """边预测器训练器"""

    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 初始化模型（使用更大的隐藏层）
        hidden_dim = getattr(config, 'HIDDEN_DIM', 1024)
        self.model = ImprovedEdgePredictor(input_dim=FEATURE_DIM, hidden_dim=hidden_dim).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 模型参数: {trainable_params:,} / {total_params:,} (可训练/总计)")

        # 优化器（使用AdamW + weight decay）
        lr = getattr(config, 'LEARNING_RATE', 0.001)
        weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 学习率调度器（余弦退火）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 混合精度训练
        self.use_amp = getattr(config, 'USE_AMP', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"⚡ 启用混合精度训练 (AMP)")

        # 梯度累积
        self.gradient_accumulation = getattr(config, 'GRADIENT_ACCUMULATION', 1)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_auc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        self.best_auc = 0.0
        self.best_epoch = 0

    def load_data(self):
        """加载预处理后的数据"""
        print("\n📊 加载数据...")

        # 加载边
        edges_train = np.load(os.path.join(PPI_PROCESSED_DIR, "edges_train.npy"))
        edges_val = np.load(os.path.join(PPI_PROCESSED_DIR, "edges_val.npy"))

        # 加载标签
        labels_train = np.load(os.path.join(PPI_PROCESSED_DIR, "labels_train.npy"))
        labels_val = np.load(os.path.join(PPI_PROCESSED_DIR, "labels_val.npy"))

        # 加载特征
        features = np.load(os.path.join(PPI_PROCESSED_DIR, "features.npy"))

        print(f"✅ 数据加载完成:")
        print(f"   • 训练边: {len(edges_train):,}")
        print(f"     └─ 正: {(labels_train==1).sum():,}, 负: {(labels_train==0).sum():,}")
        print(f"   • 验证边: {len(edges_val):,}")
        print(f"     └─ 正: {(labels_val==1).sum():,}, 负: {(labels_val==0).sum():,}")
        print(f"   • 特征: {features.shape}")

        return edges_train, labels_train, edges_val, labels_val, features

    def create_dataloader(self, edges, labels, features, shuffle=True):
        """创建数据加载器"""
        src_feats = torch.tensor(features[edges[:, 0]], dtype=torch.float32)
        dst_feats = torch.tensor(features[edges[:, 1]], dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(src_feats, dst_feats, labels_tensor)

        # 使用更多的workers以充分利用CPU
        num_workers = getattr(config, 'NUM_WORKERS', 16)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )

        return dataloader

    def train_epoch(self, dataloader):
        """训练一个epoch（支持混合精度 + 梯度累积）"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc="训练")
        for batch_idx, (src_feat, dst_feat, labels) in enumerate(pbar):
            src_feat = src_feat.to(self.device, non_blocking=True)
            dst_feat = dst_feat.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(src_feat, dst_feat).squeeze()  # [batch, 1] -> [batch]

                # BCELoss在autocast之外计算（因为模型已有Sigmoid）
                loss = self.criterion(predictions.float(), labels)
                loss = loss / self.gradient_accumulation  # 梯度累积归一化

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度累积
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # 标准训练
                predictions = self.model(src_feat, dst_feat).squeeze()  # [batch, 1] -> [batch]
                loss = self.criterion(predictions, labels)
                loss = loss / self.gradient_accumulation

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, dataloader):
        """验证（支持混合精度）"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for src_feat, dst_feat, labels in tqdm(dataloader, desc="验证", leave=False):
                src_feat = src_feat.to(self.device, non_blocking=True)
                dst_feat = dst_feat.to(self.device, non_blocking=True)

                # 混合精度推理
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(src_feat, dst_feat).squeeze()  # [batch, 1] -> [batch]
                else:
                    predictions = self.model(src_feat, dst_feat).squeeze()  # [batch, 1] -> [batch]

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算指标
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        prec = precision_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)
        rec = recall_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)

        return auc, acc, prec, rec, f1

    def train(self, edges_train, labels_train, edges_val, labels_val, features):
        """完整训练流程"""
        print(f"\n🚀 开始训练...")
        print(f"   • Epochs: {EPOCHS}")
        print(f"   • Batch Size: {BATCH_SIZE}")
        print(f"   • Learning Rate: 0.001")

        # 创建数据加载器
        train_loader = self.create_dataloader(edges_train, labels_train, features, shuffle=True)
        val_loader = self.create_dataloader(edges_val, labels_val, features, shuffle=False)

        print(f"\n📊 数据加载器:")
        print(f"   • 训练批次: {len(train_loader)}")
        print(f"   • 验证批次: {len(val_loader)}")

        # 训练循环
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # 验证
            val_auc, val_acc, val_prec, val_rec, val_f1 = self.validate(val_loader)
            self.history['val_auc'].append(val_auc)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_f1'].append(val_f1)

            epoch_time = time.time() - epoch_start

            # 打印结果
            print(f"\nEpoch {epoch}/{EPOCHS} [{epoch_time:.1f}s]")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

            # 保存最佳模型
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.save_model("best")
                print(f"  💾 保存最佳模型 (AUC={val_auc:.4f}) ⭐")

            # 更新学习率
            self.scheduler.step()

            # Early stopping（增加到15轮，因为训练轮数增加了）
            if epoch - self.best_epoch > 15:
                print(f"\n⏹️  Early stopping (无改善已15轮)")
                break

        print(f"\n✅ 训练完成!")
        print(f"   • 最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")

        return self.history

    def save_model(self, name="best"):
        """保存模型"""
        model_dir = os.path.join(current_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"edge_predictor_{name}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
        }, model_path)

    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(current_dir, "results", "training_history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"💾 训练历史已保存: {history_file}")


def main():
    print("🚀 步骤5: 训练边预测器 (真实数据 + ESM2特征)")
    print("=" * 70)

    # 创建训练器
    trainer = EdgePredictorTrainer()

    # 加载数据
    edges_train, labels_train, edges_val, labels_val, features = trainer.load_data()

    # 训练
    history = trainer.train(edges_train, labels_train, edges_val, labels_val, features)

    # 保存历史
    trainer.save_history()

    print("\n" + "=" * 70)
    print("✅ 步骤5完成: 边预测器训练完成")
    print(f"📁 模型位置: models/edge_predictor_best.pth")
    print(f"📊 最佳AUC: {trainer.best_auc:.4f}")

    if trainer.best_auc > 0.65:
        print("\n🎉 训练成功! AUC显著高于随机基线(0.50)")
    else:
        print("\n⚠️  AUC较低，可能需要调整超参数或检查数据")

    print("\n👉 下一步: 评估模型")
    print("   运行: python 6_evaluate_model.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
