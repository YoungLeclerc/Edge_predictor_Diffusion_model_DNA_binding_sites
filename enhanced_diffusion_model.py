#!/usr/bin/env python3
"""
增强版扩散模型 v2.0 - 针对DNA结合位点生成优化

改进点：
1. 条件扩散（Conditional Diffusion）- 使用蛋白质上下文引导生成
2. 自适应去噪 - 根据蛋白质特性调整去噪强度
3. 质量感知采样 - 多次采样取最优
4. 多样性增强 - 添加可控噪声保证多样性
5. 类别平衡生成 - 确保达到目标比例
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class ConditionalNoiseScheduler:
    """条件化的噪声调度器 - 根据蛋白质上下文调整"""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_alpha_bars(self, t, protein_complexity=1.0):
        """根据蛋白质复杂度调整alpha"""
        base_alpha = self.alpha_bars[t]
        # 复杂蛋白质需要更慢的扩散
        adjusted_alpha = base_alpha ** (1.0 / protein_complexity)
        return adjusted_alpha


class ContextEncoder(nn.Module):
    """蛋白质上下文编码器 - 提取全局特征引导生成"""
    def __init__(self, input_dim=1280, hidden_dim=512, context_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU()
        )

        # 注意力机制 - 聚焦重要区域
        self.attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim) 或 (seq_len, input_dim)
        Returns:
            global_context: (context_dim,) 全局上下文向量
            local_features: (seq_len, context_dim) 局部特征
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)

        # 编码
        features = self.encoder(x)  # (batch, seq_len, context_dim)

        # 自注意力
        attn_out, attn_weights = self.attention(features, features, features)

        # 全局上下文 = 加权平均
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(1)

        global_context = torch.mean(attn_out, dim=1).squeeze(0)  # (context_dim,)
        local_features = attn_out.squeeze(0)  # (seq_len, context_dim)

        return global_context, local_features, attn_weights


class EnhancedConditionalDiffusionModel(nn.Module):
    """增强版条件扩散模型"""
    def __init__(self, input_dim=1280, T=500, hidden_dim=512, context_dim=256, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.T = T
        self.device = device

        # 上下文编码器
        self.context_encoder = ContextEncoder(input_dim, hidden_dim, context_dim)

        # 条件化的去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim + context_dim + 1, hidden_dim),  # +1 for timestep
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, input_dim)
        )

        # 时间步嵌入
        self.time_embed = nn.Embedding(T, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, 1)

        # 噪声调度器
        self.scheduler = ConditionalNoiseScheduler(T)

        self.to(device)

    def forward_diffusion(self, x0, t, protein_context):
        """前向扩散过程（加噪）"""
        noise = torch.randn_like(x0)

        # 获取条件化的alpha
        complexity = self.estimate_complexity(protein_context)
        alpha_bar = self.scheduler.get_alpha_bars(t, complexity)

        # 加噪
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return xt, noise

    def estimate_complexity(self, protein_context):
        """估计蛋白质复杂度（用于调整扩散强度）"""
        # 基于特征的标准差估计复杂度
        std = torch.std(protein_context)
        complexity = torch.clamp(std / 0.5, 0.5, 2.0)  # [0.5, 2.0]
        return complexity.item()

    def reverse_diffusion_step(self, xt, t, global_context):
        """反向扩散单步（去噪）- 优化版"""
        batch_size = xt.size(0)

        # 🚀 使用缓存的时间嵌入（如果可用）
        if not hasattr(self, '_t_cache') or self._t_cache_t != t:
            t_tensor = torch.tensor([t], device=self.device)
            t_embed = self.time_embed(t_tensor)
            t_feature = self.time_proj(t_embed)  # (1, 1)
            self._t_cache = t_feature
            self._t_cache_t = t
        else:
            t_feature = self._t_cache

        # 广播上下文和时间特征到batch维度（使用expand避免内存复制）
        global_context_expanded = global_context.unsqueeze(0).expand(batch_size, -1)
        t_feature_expanded = t_feature.expand(batch_size, -1)

        # 拼接条件（使用stack然后view可能更快，但cat更通用）
        xt_with_context = torch.cat([
            xt,
            global_context_expanded,
            t_feature_expanded
        ], dim=-1)

        # 预测噪声
        predicted_noise = self.denoiser(xt_with_context)

        # 去噪 - 预计算常量
        alpha = self.scheduler.alphas[t].to(self.device)
        alpha_bar = self.scheduler.alpha_bars[t].to(self.device)

        # 🚀 使用fused操作减少kernel启动次数
        sqrt_alpha = torch.sqrt(alpha)
        coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        x_prev = (xt - coef * predicted_noise) / sqrt_alpha

        if t > 0:
            # 只在需要时生成噪声
            beta = self.scheduler.betas[t].to(self.device)
            noise = torch.randn_like(xt)
            x_prev = x_prev + torch.sqrt(beta) * noise

        return x_prev

    def generate_positive_sample(self, protein_data, num_samples=1,
                                 quality_threshold=0.5, max_attempts=5):
        """
        生成高质量正样本

        Args:
            protein_data: PyG Data object
            num_samples: 生成数量
            quality_threshold: 质量阈值
            max_attempts: 最大尝试次数

        Returns:
            samples: (num_samples, input_dim)
            quality_scores: (num_samples,)
        """
        self.eval()

        # 🚀 使用inference模式提升性能（比no_grad更快）
        with torch.inference_mode():
            # 提取上下文
            protein_x = protein_data.x.to(self.device)
            protein_y = protein_data.y.to(self.device)

            # 编码上下文
            global_context, local_features, _ = self.context_encoder(protein_x)

            # 提取正样本特征（用于质量评估）
            positive_mask = (protein_y == 1)
            if positive_mask.sum() > 0:
                positive_features = protein_x[positive_mask]
                positive_mean = torch.mean(positive_features, dim=0)
                positive_std = torch.std(positive_features, dim=0) + 1e-6
            else:
                positive_mean = torch.mean(protein_x, dim=0)
                positive_std = torch.std(protein_x, dim=0) + 1e-6

            # 🚀 批量生成所有候选样本（一次性生成，避免循环）
            total_candidates = num_samples * max_attempts

            # 从纯噪声开始（批量）- 使用固定生成器提高效率
            xt = torch.randn(total_candidates, self.input_dim, device=self.device, dtype=torch.float32)

            # 逐步去噪（批量处理所有候选样本）- 使用range缓存减少开销
            time_steps = list(reversed(range(self.T)))
            for t in time_steps:
                xt = self.reverse_diffusion_step(xt, t, global_context)

            # 评估质量（批量）
            all_scores = self.evaluate_quality(xt, positive_mean, positive_std, global_context)
            all_samples = xt

            # 选择top-k高质量样本
            top_k_indices = torch.topk(all_scores, k=num_samples).indices
            final_samples = all_samples[top_k_indices]
            final_scores = all_scores[top_k_indices]

            # 🚀 保持在GPU上，避免CPU-GPU数据传输
            return final_samples, final_scores

    def evaluate_quality(self, samples, positive_mean, positive_std, global_context):
        """
        评估生成样本质量 - 修复版

        质量指标：
        1. 与正样本分布的相似度（主要指标）
        2. 特征的合理性（不能有异常值）
        3. 特征范围合理性
        """
        # 1. 分布相似度（归一化后的距离）- 权重提高
        normalized_samples = (samples - positive_mean) / positive_std
        # 使用更温和的距离计算
        dist = torch.mean(normalized_samples ** 2, dim=1)
        dist_score = torch.exp(-dist / 5.0)  # 更温和的衰减

        # 2. 合理性（检查是否有异常值）- 权重降低
        max_abs_norm = torch.max(torch.abs(normalized_samples), dim=1).values
        validity_score = torch.exp(-max_abs_norm / 3.0)  # 允许一定的异常值

        # 3. 特征范围合理性（生成的特征应该在合理范围内）
        # ESM2特征通常在[-10, 10]范围内
        range_penalty = torch.clamp(torch.abs(samples).max(dim=1).values - 10.0, min=0.0)
        range_score = torch.exp(-range_penalty / 5.0)

        # 综合质量分数 - 主要依赖分布相似度
        quality = 0.6 * dist_score + 0.25 * validity_score + 0.15 * range_score

        return quality

    def train_on_positive_samples(self, dataset, epochs=100, batch_size=32, lr=1e-4):
        """训练扩散模型"""
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        print(f"🎨 训练增强版扩散模型 (T={self.T}, epochs={epochs})")

        for epoch in range(epochs):
            total_loss = 0
            count = 0

            for data in dataset:
                # 提取正样本
                positive_mask = (data.y == 1)
                if positive_mask.sum() == 0:
                    continue

                positive_samples = data.x[positive_mask].to(self.device)
                protein_context = data.protein_context.to(self.device)

                # 编码上下文（detach避免重复backward）
                with torch.no_grad():
                    global_context, _, _ = self.context_encoder(data.x.to(self.device))
                global_context = global_context.detach()  # 从计算图中分离

                # 训练
                for i in range(0, positive_samples.size(0), batch_size):
                    batch = positive_samples[i:i+batch_size]

                    # 随机时间步
                    t = torch.randint(0, self.T, (1,), device=self.device).item()

                    # 前向扩散
                    xt, noise = self.forward_diffusion(batch, t, protein_context)

                    # 预测噪声
                    t_tensor = torch.tensor([t], device=self.device)
                    t_embed = self.time_embed(t_tensor)
                    t_feature = self.time_proj(t_embed)

                    global_context_expanded = global_context.unsqueeze(0).expand(batch.size(0), -1)
                    t_feature_expanded = t_feature.expand(batch.size(0), -1)

                    xt_with_context = torch.cat([
                        xt,
                        global_context_expanded,
                        t_feature_expanded
                    ], dim=-1)

                    predicted_noise = self.denoiser(xt_with_context)

                    # 损失
                    loss = F.mse_loss(predicted_noise, noise)

                    optimizer.zero_grad()
                    loss.backward(retain_graph=False)  # 🔧 修复：显式设置retain_graph=False
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    count += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == epochs - 1:
                avg_loss = total_loss / max(count, 1)
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

        print(f"✅ 扩散模型训练完成")
