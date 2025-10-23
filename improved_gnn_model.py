#!/usr/bin/env python3
"""
改进的GNN模型 - 针对测试性能优化
包含Focal Loss、更强正则化、概率校准等技术
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_recall_curve, auc,
                             accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, balanced_accuracy_score)
import numpy as np
import os


class ImprovedResidualBlock(nn.Module):
    """改进的残差块 - 增强正则化"""
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout * 0.5)  # 第二层dropout稍微低一点
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))


class ImprovedBindingSiteGNN(nn.Module):
    """改进的绑定位点GNN - 针对测试性能优化"""
    
    def __init__(self, input_dim=1280, hidden_dim=256, dropout=0.5, 
                 use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.5):
        super().__init__()
        
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 输入投影层 - 增加正则化
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)  # 输入层稍微低一点的dropout
        )

        # 图卷积层 - 确保所有输出维度一致
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)  # 单头GAT，输出维度=hidden_dim
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        # 残差块 - 减少层数
        self.res_blocks = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(1)  # 只用1个残差块
        ])

        # 输出层 - 简化结构
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # 减少隐藏层大小
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),  # 增加一层但减少参数
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

        # 损失函数
        if use_focal_loss:
            # 不使用pos_weight，让Focal Loss自动处理
            self.loss_fn = self.focal_loss
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
        # 温度参数用于概率校准
        self.temperature = nn.Parameter(torch.ones(1))

    def focal_loss(self, pred, target):
        """Focal Loss实现"""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 输入投影
        x = self.input_proj(x)

        # 🔧 修复：验证和清理边索引
        num_nodes = x.size(0)

        if edge_index.numel() == 0:
            # 空边列表 - 创建自循环
            edge_index = torch.arange(num_nodes, device=x.device, dtype=torch.long)
            edge_index = torch.stack([edge_index, edge_index], dim=0)
        else:
            # 检查边的有效性
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                # 移除超出范围的边
                valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]

                # 如果没有有效边，创建自循环
                if edge_index.numel() == 0:
                    edge_index = torch.arange(num_nodes, device=x.device, dtype=torch.long)
                    edge_index = torch.stack([edge_index, edge_index], dim=0)

        # 多类型图卷积 - 使用residual connection
        identity = x
        x1 = F.elu(self.conv1(x, edge_index))
        x2 = F.elu(self.conv2(x, edge_index))
        x3 = F.elu(self.conv3(x, edge_index))
        x = x1 + x2 + x3 + identity  # 添加输入residual

        # 残差块
        for block in self.res_blocks:
            x = block(x)

        # 分类输出
        logits = self.classifier(x).squeeze()
        
        # 训练时返回原始logits，推理时使用温度校准
        if self.training:
            return logits
        else:
            return logits / self.temperature  # 温度校准

    def train_model(self, train_data, val_data, epochs=30, lr=5e-4, device='cpu', patience=5):
        """训练模型 - 改进版"""
        self.to(device)
        
        # 使用AdamW优化器，增加权重衰减
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=1e-3,  # 增加权重衰减
            betas=(0.9, 0.999)
        )
        
        # 使用更激进的学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=3
        )

        best_val_auc = 0
        best_val_f1 = 0
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            batch_count = 0

            for data in train_data:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                optimizer.zero_grad()
                out = self(data)

                if (data.y == 1).sum().item() == 0:
                    continue

                loss = self.loss_fn(out, data.y.float())
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)  # 更激进的梯度裁剪

                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0

            # 验证
            val_metrics = self.evaluate(val_data, device)
            val_f1 = val_metrics['f1']
            val_auc_pr = val_metrics['auc_pr']

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val AUC-PR: {val_auc_pr:.4f} | "
                  f"Val ACC: {val_metrics['accuracy']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Temp: {self.temperature.item():.3f}")

            # 学习率调度
            scheduler.step(val_auc_pr)

            # 保存最佳模型
            if val_auc_pr > best_val_auc:
                best_val_auc = val_auc_pr
                best_val_f1 = val_f1
                torch.save(self.state_dict(), "best_improved_gnn_model.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if os.path.exists("best_improved_gnn_model.pt"):
            self.load_state_dict(torch.load("best_improved_gnn_model.pt"))
            
        print(f"Training complete. Best Val AUC-PR: {best_val_auc:.4f}, Best Val F1: {best_val_f1:.4f}")
        return best_val_auc, best_val_f1

    def evaluate(self, dataset, device='cpu'):
        """评估函数 - 包含概率校准"""
        if not dataset:
            return {'f1': 0, 'mcc': 0, 'auc_pr': 0, 'accuracy': 0, 'balanced_accuracy': 0,
                   'precision': 0, 'recall': 0, 'specificity': 0, 'auc_roc': 0}

        self.eval()
        self.to(device)
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for data in dataset:
                if data.x.size(0) == 0:
                    continue

                data = data.to(device)
                out = self(data)  # 自动应用温度校准
                probs = torch.sigmoid(out)

                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(data.y.cpu().tolist())

        if len(all_labels) == 0:
            return {'f1': 0, 'mcc': 0, 'auc_pr': 0, 'accuracy': 0, 'balanced_accuracy': 0,
                   'precision': 0, 'recall': 0, 'specificity': 0, 'auc_roc': 0}

        all_labels = [int(label) for label in all_labels]
        
        # 寻找最优阈值 - 针对不同指标
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_metrics = {
            'f1': {'value': 0, 'threshold': 0.5},
            'balanced_acc': {'value': 0, 'threshold': 0.5}
        }
        
        for threshold in thresholds:
            preds = [1 if p > threshold else 0 for p in all_probs]
            
            f1 = f1_score(all_labels, preds, zero_division=0)
            balanced_acc = balanced_accuracy_score(all_labels, preds)
            
            if f1 > best_metrics['f1']['value']:
                best_metrics['f1']['value'] = f1
                best_metrics['f1']['threshold'] = threshold
                
            if balanced_acc > best_metrics['balanced_acc']['value']:
                best_metrics['balanced_acc']['value'] = balanced_acc
                best_metrics['balanced_acc']['threshold'] = threshold

        # 使用平衡准确率最优阈值进行最终评估
        best_threshold = best_metrics['balanced_acc']['threshold']
        final_preds = [1 if p > best_threshold else 0 for p in all_probs]
        
        # 计算所有指标
        accuracy = accuracy_score(all_labels, final_preds)
        balanced_acc = balanced_accuracy_score(all_labels, final_preds)
        precision = precision_score(all_labels, final_preds, zero_division=0)
        recall = recall_score(all_labels, final_preds, zero_division=0)
        f1 = f1_score(all_labels, final_preds, zero_division=0)
        mcc = matthews_corrcoef(all_labels, final_preds)
        
        tn, fp, fn, tp = confusion_matrix(all_labels, final_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC指标
        auc_pr = float('nan')
        auc_roc = float('nan')
        
        if any(label == 1 for label in all_labels):
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                auc_pr = auc(recall_curve, precision_curve)
                auc_roc = roc_auc_score(all_labels, all_probs)
            except:
                pass

        return {
            'f1': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': recall,
            'mcc': mcc,
            'auc_pr': auc_pr,
            'auc_roc': auc_roc,
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'best_threshold_f1': best_metrics['f1']['threshold'],
            'best_threshold_balanced_acc': best_metrics['balanced_acc']['threshold'],
            'total_samples': len(all_labels),
            'positive_samples': sum(all_labels),
            'negative_samples': len(all_labels) - sum(all_labels),
            'positive_ratio': sum(all_labels) / len(all_labels) if len(all_labels) > 0 else 0
        }