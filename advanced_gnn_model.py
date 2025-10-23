#!/usr/bin/env python3
"""
高级GNN模型 v2.0 - DNA结合位点预测专用

改进点：
1. Graph Attention Networks (GAT) - 自适应邻居权重
2. 残差连接 + 层归一化 - 更深网络，更好梯度流
3. 边特征融合 - 利用边预测器提供的边权重
4. 多尺度特征聚合 - 局部+全局信息
5. 不平衡数据专用损失 - Focal Loss + Class Balanced Loss
6. 自适应正样本权重 - 动态调整样本权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import degree
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, accuracy_score
import numpy as np


class MultiScaleGATLayer(nn.Module):
    """多尺度图注意力层"""
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()

        # 多头注意力
        self.gat = GATv2Conv(
            in_dim,
            out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=1,  # 支持边特征
            concat=True
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(out_dim)

        # 残差投影（如果维度不匹配）
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, 1) 边权重/特征
        """
        # 残差连接
        identity = x
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)

        # GAT
        x = self.gat(x, edge_index, edge_attr=edge_attr)

        # 残差 + LayerNorm + Dropout
        x = self.layer_norm(x + identity)
        x = self.dropout(x)

        return x


class AdvancedBindingSiteGNN(nn.Module):
    """高级DNA结合位点预测GNN"""
    def __init__(
        self,
        input_dim=1280,
        hidden_dim=256,
        num_layers=4,
        heads=4,
        dropout=0.3,
        use_edge_features=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_balanced=True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_balanced = class_balanced

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多层GAT
        self.gat_layers = nn.ModuleList([
            MultiScaleGATLayer(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 全局信息聚合
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 局部-全局融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # local + global
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 输出头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 边特征编码器（如果使用边特征）
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, data):
        """
        Args:
            data: PyG Data object
        Returns:
            logits: (num_nodes,)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 边特征
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None

        # 输入投影
        x = self.input_proj(x)  # (num_nodes, hidden_dim)

        # 多层GAT（局部信息）
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)

        # 全局信息
        global_mean = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        global_max = global_max_pool(x, batch)    # (num_graphs, hidden_dim)
        global_info = torch.cat([global_mean, global_max], dim=1)  # (num_graphs, hidden_dim*2)
        global_info = self.global_pool(global_info)  # (num_graphs, hidden_dim)

        # 广播全局信息到所有节点
        global_info_expanded = global_info[batch]  # (num_nodes, hidden_dim)

        # 局部-全局融合
        x_fused = torch.cat([x, global_info_expanded], dim=1)  # (num_nodes, hidden_dim*2)
        x = self.fusion(x_fused)  # (num_nodes, hidden_dim)

        # 分类
        logits = self.classifier(x).squeeze(-1)  # (num_nodes,)

        return logits

    def compute_loss(self, logits, targets, effective_num=None):
        """
        计算损失（Focal Loss + Class Balanced）

        Args:
            logits: (num_nodes,)
            targets: (num_nodes,) 0/1
            effective_num: dict with 'num_pos', 'num_neg' for class balancing
        """
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction='none'
        )

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma

        alpha_t = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)
        focal_loss = alpha_t * focal_weight * bce_loss

        # Class Balanced Loss (可选)
        if self.class_balanced and effective_num is not None:
            num_pos = effective_num.get('num_pos', 1)
            num_neg = effective_num.get('num_neg', 1)
            beta = 0.9999  # 超参数

            # Effective number
            effective_num_pos = (1 - beta ** num_pos) / (1 - beta)
            effective_num_neg = (1 - beta ** num_neg) / (1 - beta)

            # Class weights
            weight_pos = effective_num_neg / (effective_num_pos + effective_num_neg)
            weight_neg = effective_num_pos / (effective_num_pos + effective_num_neg)

            # 应用权重
            class_weights = targets * weight_pos + (1 - targets) * weight_neg
            focal_loss = focal_loss * class_weights

        return focal_loss.mean()

    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3,
                   device='cuda', patience=15, save_path=None):
        """训练模型"""
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        best_f1 = 0
        patience_counter = 0

        print(f"🚀 开始训练 Advanced GNN (layers={self.num_layers}, hidden={self.hidden_dim})")

        for epoch in range(epochs):
            # 训练
            self.train()
            total_loss = 0
            num_batches = 0

            # 统计类别数量（用于class balanced loss）
            total_pos = 0
            total_neg = 0

            for batch_data in train_loader:
                batch_data = batch_data.to(device)

                # 统计
                total_pos += (batch_data.y == 1).sum().item()
                total_neg += (batch_data.y == 0).sum().item()

                # 前向
                logits = self(batch_data)
                loss = self.compute_loss(
                    logits,
                    batch_data.y,
                    effective_num={'num_pos': total_pos, 'num_neg': total_neg}
                )

                # 反向
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            # 验证
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_metrics = self.evaluate(val_loader, device)
                scheduler.step(val_metrics['f1'])

                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                      f"Val F1: {val_metrics['f1']:.4f} - "
                      f"Val MCC: {val_metrics['mcc']:.4f} - "
                      f"Val AUC-PR: {val_metrics['auc_pr']:.4f}")

                # 早停
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    patience_counter = 0

                    if save_path:
                        torch.save(self.state_dict(), save_path)
                        print(f"  💾 模型已保存 (F1={best_f1:.4f})")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"  ⏹️  早停 (patience={patience})")
                    break

        print(f"✅ 训练完成 - 最佳F1: {best_f1:.4f}")
        return best_f1

    def evaluate(self, data_loader, device='cuda'):
        """评估模型"""
        self.eval()
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                logits = self(batch_data)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(batch_data.y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # 计算指标
        metrics = {
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'mcc': matthews_corrcoef(all_targets, all_preds),
            'accuracy': accuracy_score(all_targets, all_preds),
            'auc_pr': average_precision_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0,
            'auc_roc': roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0
        }

        return metrics
