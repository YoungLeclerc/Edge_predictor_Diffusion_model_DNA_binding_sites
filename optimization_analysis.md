# GPU利用率低的原因分析与优化方案

## 🔍 问题诊断

当前代码的性能瓶颈：

### 1. 边预测器构建图（CPU密集）
```python
# 当前实现（逐个处理，CPU计算多）
for test_data in test_dataset:
    augmented_graph = build_edges_with_edge_predictor(
        test_data, ..., device=config.device
    )
```

**问题**：
- 逐个蛋白质处理（串行）
- 余弦相似度、距离计算在CPU
- Top-K选择在CPU
- 数据在CPU和GPU之间频繁传输

### 2. 数据加载（ESM2特征提取）
```python
# ProteinDataset 在初始化时提取ESM2特征
dataset_loader = ProteinDataset(temp_dir, device=config.device)
```

**问题**：
- ESM2模型可能在CPU运行
- 逐个序列处理
- 没有批量化

### 3. 边预测器推理
```python
# build_edges_with_edge_predictor 内部
pred_scores = edge_predictor(xi, xj)  # 可能在CPU
```

**问题**：
- 如果edge_predictor没有正确放到GPU
- 或者输入数据没有放到GPU

## 🚀 优化方案

### 方案1：批量化边预测（推荐）⭐

将逐个蛋白质处理改为批量处理：

```python
def build_edges_batch_gpu(dataset, edge_predictor, config, batch_size=4):
    """批量在GPU上构建边"""
    augmented_dataset = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]

        # 批量处理
        batch_graphs = []
        for data in batch:
            # 确保数据在GPU
            data = data.to(config.device)

            # GPU上计算
            with torch.no_grad():
                graph = build_edges_with_edge_predictor_gpu(
                    data, edge_predictor, config
                )
            batch_graphs.append(graph)

        augmented_dataset.extend(batch_graphs)

    return augmented_dataset
```

### 方案2：优化edge_predictor内部计算

在`edge_predictor_augmentation.py`中优化：

```python
def build_edges_with_edge_predictor_gpu(original_data, edge_predictor, config):
    """GPU优化版本"""
    device = config.device

    # 1. 确保所有张量在GPU
    x = original_data.x.to(device)

    # 2. GPU上批量计算边预测
    with torch.no_grad():
        # 一次性计算所有节点对的预测分数
        num_nodes = x.size(0)

        # 使用矩阵乘法加速（而非循环）
        pred_scores = compute_edge_scores_batch(x, edge_predictor, device)

        # GPU上计算余弦相似度（向量化）
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.t())  # GPU矩阵乘法

        # GPU上Top-K选择
        top_k_indices = torch.topk(pred_scores, k=config.top_k, dim=1).indices

    # 3. 构建边索引（在GPU）
    edge_index = build_edge_index_from_topk_gpu(top_k_indices, device)

    return Data(x=x, edge_index=edge_index, y=original_data.y)
```

### 方案3：数据预加载和缓存

```python
class GPUDataLoader:
    """GPU数据预加载器"""
    def __init__(self, dataset, device, prefetch=2):
        self.dataset = dataset
        self.device = device
        self.prefetch = prefetch

    def __iter__(self):
        # 使用多线程预加载下一批数据到GPU
        for data in self.dataset:
            # 异步传输到GPU
            data_gpu = data.to(self.device, non_blocking=True)
            yield data_gpu
```

### 方案4：减少边预测器的使用

**最简单有效的方案**：

```python
# 配置中设置更严格的阈值，减少计算量
self.edge_predictor_config = {
    'predictor_threshold': 0.95,  # 极严格，大部分边直接跳过
    'sim_threshold': 0.9,
    'top_k': 5,
    'use_topk_guarantee': True  # 只保证top-k，其他不计算
}
```

## 📊 优化优先级

1. **立即可做**（5分钟）：
   - 检查edge_predictor是否在GPU：`edge_predictor.to(device)`
   - 提高阈值减少计算：`predictor_threshold=0.95`

2. **短期优化**（30分钟）：
   - 确保所有张量操作在GPU
   - 添加`torch.cuda.synchronize()`检查同步点

3. **长期优化**（2小时）：
   - 重写batch版本的边构建函数
   - 向量化相似度和距离计算

## 🔧 快速修复清单

```python
# 1. 确保edge_predictor在GPU
edge_predictor = edge_predictor.to(config.device)
edge_predictor.eval()

# 2. 确保数据在GPU
data = data.to(config.device)

# 3. 使用torch.no_grad()减少内存
with torch.no_grad():
    predictions = edge_predictor(x)

# 4. 避免频繁的CPU-GPU传输
# ❌ 不好
for i in range(n):
    x_cpu = x[i].cpu().numpy()  # 每次都传输

# ✅ 好
x_cpu = x.cpu().numpy()  # 一次性传输
for i in range(n):
    xi = x_cpu[i]
```
