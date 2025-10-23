# PPI边预测器构建平衡图结构 - 详细说明

## 🎯 核心流程

### 1. 生成正样本节点 (扩散模型)

```python
# robust_augment_dataset_with_edge_predictor() 第370行
candidate_samples = diffusion_model.generate_positive_sample(
    protein_context,
    num_samples=n_to_generate * 3  # 生成候选
)
```

**输出**: 新的正样本节点特征 (shape: [N, 1280])

---

### 2. 质量控制与多样性筛选

```python
# 第381-393行
quality_samples, quality_score = calculate_sample_quality(
    candidate_samples, real_pos_samples, threshold=0.7
)

diverse_samples, diversity_score = calculate_sample_diversity(
    quality_samples, threshold=0.3
)
```

**输出**: 高质量、多样化的正样本节点

---

### 3. 🔑 使用PPI边预测器构建图结构

```python
# 第402-409行
augmented_graph, edge_info = build_edges_with_edge_predictor(
    data,              # 原始图 (Data对象)
    new_x,             # 生成的正样本节点
    edge_predictor,    # 训练好的PPI边预测器 (AUC=0.9019)
    config.device,
    predictor_threshold=0.5,    # 边预测概率阈值
    sim_threshold=0.6,          # 余弦相似度阈值
    dist_threshold=1.5,         # 欧氏距离阈值
    top_k=5,                    # Top-K保证最小连接数
    connect_generated_nodes=True,  # 是否连接生成节点之间的边
    use_topk_guarantee=True     # 使用Top-K保证策略
)
```

**输出**: 完整的平衡图结构 (新节点 + 智能边)

---

## 📊 边构建详细策略

### 第1部分: 生成节点 ↔ 原始节点的边

```python
# build_edges_with_edge_predictor() 第128-198行

for 每个新生成的节点 i:
    for 每个原始节点 j:
        # 1. PPI边预测器打分
        ppi_score = edge_predictor(node_i, node_j)  # [0, 1]

        # 2. 余弦相似度
        cos_sim = cosine_similarity(node_i, node_j)  # [-1, 1]

        # 3. 欧氏距离
        euclidean_dist = ||node_i - node_j||_2

        # 4. 混合决策
        if use_topk_guarantee:
            # 策略A: 满足条件 OR Top-K
            if (ppi_score > 0.5 AND cos_sim > 0.6 AND dist < 1.5):
                连接 (i, j)
            OR
            if j in Top-5(ppi_scores):
                连接 (i, j)  # 保证最小连接度
        else:
            # 策略B: 严格条件
            if (ppi_score > 0.5 AND cos_sim > 0.6 AND dist < 1.5):
                连接 (i, j)
```

**关键优势**:
- ✅ **PPI知识驱动**: 基于1,858,944条真实PPI数据训练
- ✅ **混合评估**: 3个指标综合判断
- ✅ **Top-K保证**: 避免孤立节点，确保图连通性

---

### 第2部分: 生成节点 ↔ 生成节点的边（可选）

```python
# 第206-228行

if connect_generated_nodes:
    for 生成节点 i:
        for 生成节点 j (j > i):
            ppi_score = edge_predictor(node_i, node_j)
            cos_sim = cosine_similarity(node_i, node_j)
            dist = ||node_i - node_j||_2

            if (ppi_score > 0.5 AND cos_sim > 0.6 AND dist < 1.5):
                连接 (i, j)
```

**作用**: 增加生成节点之间的相互作用，增强图结构

---

## 🎯 最终生成的平衡图

### 图结构组成

```python
augmented_graph = Data(
    x = [原始节点特征; 生成的正样本节点特征],  # shape: [N_orig + N_gen, 1280]
    y = [原始节点标签; 全1标签(正样本)],        # shape: [N_orig + N_gen]
    edge_index = [
        原始边,                                # 保留原始图的边
        新节点↔原始节点的边,                    # PPI预测器决定
        新节点↔新节点的边                       # PPI预测器决定
    ]
)
```

### 平衡性

**原始图**:
- 正样本: 50个
- 负样本: 500个
- 比例: 9.1%

**增强后的图** (target_ratio=0.9):
- 正样本: 450个 (原始50 + 生成400)
- 负样本: 50个 (下采样)
- 比例: 90%

---

## 🚀 输入到GNN训练

```python
# cross_validation_training_with_edge_predictor()
# robust_pipeline_edge.py 第223-278行

model = RobustGNNModel(input_dim=1280, hidden_dim=512)

for each fold in 3-fold CV:
    train_data = augmented_graphs + original_train_graphs
    val_data = original_val_graphs

    model.train_with_domain_adaptation(
        train_data,  # 包含平衡的图结构
        val_data,
        epochs=50
    )
```

**训练输入**: 平衡的图结构
- 节点: 原始 + 生成的正样本
- 边: 由PPI边预测器智能构建
- 标签: 0/1 (结合位点/非结合位点)

**训练目标**: 预测蛋白质序列中的DNA/RNA结合位点

---

## 🔧 参数调优指南

### 如果想要更严格的边（减少边数）

```python
config.edge_predictor_config = {
    'predictor_threshold': 0.7,    # ↑ 提高阈值
    'sim_threshold': 0.7,          # ↑ 提高相似度要求
    'dist_threshold': 1.0,         # ↓ 降低距离阈值
    'top_k': 3,                    # ↓ 减少Top-K
    'use_topk_guarantee': False    # 关闭Top-K保证
}
```

### 如果想要更连通的图（增加边数）

```python
config.edge_predictor_config = {
    'predictor_threshold': 0.3,    # ↓ 降低阈值
    'sim_threshold': 0.4,          # ↓ 降低相似度要求
    'dist_threshold': 2.0,         # ↑ 提高距离阈值
    'top_k': 10,                   # ↑ 增加Top-K
    'use_topk_guarantee': True,    # 开启Top-K保证
    'connect_generated_nodes': True # 连接生成节点间的边
}
```

### 推荐配置（默认值）

```python
config.edge_predictor_config = {
    'predictor_threshold': 0.5,    # 平衡精确度和召回率
    'sim_threshold': 0.6,          # 中等相似度要求
    'dist_threshold': 1.5,         # 中等距离要求
    'top_k': 5,                    # 保证最小连通性
    'use_topk_guarantee': True,    # 推荐开启
    'connect_generated_nodes': True # 增强图结构
}
```

---

## 📈 与其他方法的对比

| 方法 | 节点生成 | 边构建 | 优势 | 劣势 |
|------|---------|--------|------|------|
| **Random** | ❌ | 随机连接 | 快速 | 无生物学意义 |
| **KNN** | ❌ | 基于距离 | 简单 | 仅考虑特征空间距离 |
| **当前方法 (PPI)** | ✅ 扩散模型 | PPI边预测器 | 生物学准确、高性能 | 需要预训练模型 |

---

## ✅ 总结

**您的理解完全正确！**

当前实现的完整流程：
1. ✅ 扩散模型生成正样本节点特征
2. ✅ 质量控制 + 多样性筛选
3. ✅ **PPI边预测器智能构建边**（基于真实PPI知识）
4. ✅ 生成平衡的图结构（节点+边）
5. ✅ 输入到GNN进行蛋白质结合位点预测

**关键优势**:
- 🧬 **生物学准确**: 基于STRING数据库的19,488个蛋白质的1,858,944条相互作用
- 🎯 **高性能**: PPI边预测器AUC=0.9019
- 📊 **平衡训练**: 正负样本比例可控
- 🔗 **智能连边**: 混合评估机制（PPI分数+相似度+距离）
- 🛡️ **鲁棒性**: Top-K保证避免孤立节点

这正是您想要的**生成平衡图结构再进行GNN训练**的完整方案！
