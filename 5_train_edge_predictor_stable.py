#!/usr/bin/env python3
"""
步骤5: 训练边预测器（稳定版 - 使用Warmup + 更低学习率）
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

# 导入稳定配置
current_dir = os.path.dirname(os.path.abspath(__file__))
import importlib.util
config_path = os.path.join(current_dir, "ppi_config_stable.py")
spec = importlib.util.spec_from_file_location("ppi_config_stable", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

FEATURE_DIM = config.FEATURE_DIM
DEVICE = config.DEVICE
EPOCHS = config.TRAIN_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
PPI_PROCESSED_DIR = config.PPI_PROCESSED_DIR


class WarmupCosineScheduler:
    """Warmup + Cosine退火学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        """更新学习率"""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        """返回当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class StableEdgePredictorTrainer:
    """稳定版边预测器训练器"""

    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")

        # 初始化模型
        hidden_dim = getattr(config, 'HIDDEN_DIM', 1024)
        self.model = ImprovedEdgePredictor(input_dim=FEATURE_DIM, hidden_dim=hidden_dim).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 模型参数: {trainable_params:,} / {total_params:,} (可训练/总计)")

        # 优化器（稳定版参数）
        lr = getattr(config, 'LEARNING_RATE', 0.0005)
        weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)  # 更稳定的beta参数
        )

        print(f"📈 学习率配置:")
        print(f"   • 基础学习率: {lr}")
        print(f"   • Weight Decay: {weight_decay}")

        # Warmup + Cosine退火调度器
        warmup_epochs = getattr(config, 'LR_WARMUP_EPOCHS', 3)
        min_lr = getattr(config, 'LR_MIN', 1e-6)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=EPOCHS,
            base_lr=lr,
            min_lr=min_lr
        )
        print(f"   • Warmup: {warmup_epochs} epochs")
        print(f"   • 最小学习率: {min_lr}")

        # 损失函数
        self.criterion = nn.BCELoss()

        # 混合精度训练
        self.use_amp = getattr(config, 'USE_AMP', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"⚡ 启用混合精度训练 (AMP)")

        # 梯度裁剪
        self.gradient_clip_norm = getattr(config, 'GRADIENT_CLIP_NORM', 0.5)
        print(f"✂️  梯度裁剪: {self.gradient_clip_norm}")

        # 梯度累积
        self.gradient_accumulation = getattr(config, 'GRADIENT_ACCUMULATION', 1)

        # Early Stopping
        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)
        self.min_delta = getattr(config, 'MIN_DELTA', 0.0001)
        print(f"⏸️  Early Stopping: patience={self.patience}, min_delta={self.min_delta}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_auc': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }

        self.best_auc = 0.0
        self.best_epoch = 0
        self.no_improve_count = 0

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

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (src_feat, dst_feat, labels) in enumerate(pbar):
            src_feat = src_feat.to(self.device, non_blocking=True)
            dst_feat = dst_feat.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 混合精度前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(src_feat, dst_feat).squeeze()

                loss = self.criterion(predictions.float(), labels)
                loss = loss / self.gradient_accumulation

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                predictions = self.model(src_feat, dst_feat).squeeze()
                loss = self.criterion(predictions, labels)
                loss = loss / self.gradient_accumulation

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1

            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation:.4f}',
                'lr': f'{current_lr:.6f}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, dataloader):
        """验证"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for src_feat, dst_feat, labels in tqdm(dataloader, desc="验证", leave=False):
                src_feat = src_feat.to(self.device, non_blocking=True)
                dst_feat = dst_feat.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(src_feat, dst_feat).squeeze()
                else:
                    predictions = self.model(src_feat, dst_feat).squeeze()

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
        print(f"\n🚀 开始稳定训练...")
        print(f"   • Epochs: {EPOCHS}")
        print(f"   • Batch Size: {BATCH_SIZE}")
        print(f"   • Learning Rate: {config.LEARNING_RATE} (with Warmup)")

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
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # 更新学习率（Warmup + Cosine）
            current_lr = self.scheduler.step()
            self.history['learning_rate'].append(current_lr)

            # 验证
            val_auc, val_acc, val_prec, val_rec, val_f1 = self.validate(val_loader)
            self.history['val_auc'].append(val_auc)
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)
            self.history['val_f1'].append(val_f1)

            epoch_time = time.time() - epoch_start

            # 打印结果
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{EPOCHS} [{epoch_time:.1f}s] | LR: {current_lr:.6f}")
            print(f"{'='*70}")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  AUC:  {val_auc:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f}")
            print(f"  Rec:  {val_rec:.4f} | F1:  {val_f1:.4f}")

            # Early Stopping检查
            improvement = val_auc - self.best_auc

            if improvement > self.min_delta:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.no_improve_count = 0
                self.save_model("best_stable")
                print(f"  💾 保存最佳模型 (AUC={val_auc:.4f}) ⭐ [提升: +{improvement:.4f}]")
            else:
                self.no_improve_count += 1
                print(f"  ⏸️  无改善 ({self.no_improve_count}/{self.patience})")

                if self.no_improve_count >= self.patience:
                    print(f"\n⏹️  Early stopping: 已{self.patience}轮无改善 (min_delta={self.min_delta})")
                    break

        print(f"\n{'='*70}")
        print(f"✅ 训练完成!")
        print(f"{'='*70}")
        print(f"   • 最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")
        print(f"   • 总训练轮数: {epoch}")

        return self.history

    def save_model(self, name="best_stable"):
        """保存模型"""
        model_dir = os.path.join(current_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"edge_predictor_{name}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'config': {
                'learning_rate': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'hidden_dim': config.HIDDEN_DIM,
                'warmup_epochs': config.LR_WARMUP_EPOCHS
            }
        }, model_path)

    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(current_dir, "results", "training_history_stable.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"💾 训练历史已保存: {history_file}")


def main():
    print("🚀 步骤5: 训练边预测器 (稳定版 - Warmup + Lower LR)")
    print("=" * 70)

    # 创建训练器
    trainer = StableEdgePredictorTrainer()

    # 加载数据
    edges_train, labels_train, edges_val, labels_val, features = trainer.load_data()

    # 训练
    history = trainer.train(edges_train, labels_train, edges_val, labels_val, features)

    # 保存历史
    trainer.save_history()

    print("\n" + "=" * 70)
    print("✅ 步骤5完成: 边预测器稳定训练完成")
    print(f"📁 模型位置: models/edge_predictor_best_stable.pth")
    print(f"📊 最佳AUC: {trainer.best_auc:.4f}")

    if trainer.best_auc > 0.65:
        print("\n🎉 训练成功! AUC显著高于随机基线(0.50)")
    else:
        print("\n⚠️  AUC较低，可能需要调整超参数或检查数据")

    print("\n👉 下一步: 评估稳定模型")
    print("   运行: python 6_evaluate_model.py (需要修改模型路径)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
