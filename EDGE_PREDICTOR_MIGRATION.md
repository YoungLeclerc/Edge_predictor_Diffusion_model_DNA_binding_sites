# 从KNN到PPI边预测器的迁移

## 📊 背景

之前的pipeline使用**KNN (K-Nearest Neighbors)** 方法基于特征空间的欧氏距离来构建图结构。现在我们已经升级到使用**训练好的PPI边预测器**来构建更准确的图结构。

## 🔄 主要变化

### 1. **图构建方法**

**之前 (KNN)**:
```python
# 基于欧氏距离的K近邻
edge_index = create_knn_edges(features, k=config.knn_k)
```

**现在 (PPI边预测器)**:
```python
# 基于训练好的PPI模型预测边
edge_index = edge_predictor.predict_edges(
    src_features,
    dst_features,
    threshold=config.edge_predictor_config['predictor_threshold'],
    top_k=config.edge_predictor_config['top_k']
)
```

### 2. **已废弃的参数**

| 参数 | 位置 | 状态 | 说明 |
|------|------|------|------|
| `config.knn_k` | config.py:32 | ✅ 已标记废弃 | 仅保留用于向后兼容 |
| KNN打印信息 | robust_pipeline_edge.py:74 | ✅ 已移除 | 替换为Top-K保证信息 |

### 3. **新增的配置**

**边预测器配置** (`config.edge_predictor_config`):
```python
{
    'predictor_threshold': 0.5,    # 边预测概率阈值
    'sim_threshold': 0.6,          # 余弦相似度阈值
    'dist_threshold': 1.5,         # 欧氏距离阈值
    'top_k': 5,                    # 每个节点最少保留边数
    'connect_generated_nodes': True,  # 连接生成节点到原始图
    'use_topk_guarantee': True     # 使用Top-K保证机制
}
```

## 📈 性能提升

### PPI边预测器性能
- **训练数据**: STRING v12.0 人类PPI数据
- **样本量**: 1,858,944条相互作用，19,488个蛋白质
- **模型性能**: AUC = **0.9019** (优秀)
- **特征**: ESM2 1280维蛋白质嵌入

### 优势对比

| 方面 | KNN | PPI边预测器 |
|------|-----|-------------|
| **生物学准确性** | ❌ 纯距离度量 | ✅ 基于真实PPI知识 |
| **泛化能力** | ⚠️ 依赖特征空间 | ✅ 学习到蛋白质相互作用模式 |
| **可解释性** | ⚠️ 仅反映特征相似度 | ✅ 反映生物学相互作用 |
| **准确度** | ⚠️ 依赖k值选择 | ✅ AUC=0.9019 |
| **灵活性** | ❌ 固定k值 | ✅ 阈值+Top-K混合策略 |

## 🔧 代码更新

### 修改的文件

1. **robust_pipeline_edge.py**
   - ✅ 更新文档字符串，强调PPI优势
   - ✅ 移除KNN k值打印
   - ✅ 添加Top-K保证信息打印
   - ✅ 第271-280行: 自动加载PPI模型

2. **config.py**
   - ✅ 标记 `knn_k` 为废弃参数
   - ℹ️ 保留参数以避免破坏现有代码

3. **load_or_train_edge_predictor()**
   - ✅ 自动检测hidden_dim
   - ✅ PyTorch 2.6兼容性
   - ✅ 加载预训练PPI模型

## 📝 使用建议

### 推荐配置

```python
# robust_pipeline_edge.py 中的配置
config = RobustTrainingConfigWithEdgePredictor(
    use_edge_predictor=True,
    target_ratio=0.9
)

# 边预测器配置建议
config.edge_predictor_config = {
    'predictor_threshold': 0.5,   # 平衡精确度和召回率
    'top_k': 5,                   # 确保最小连接度
    'use_topk_guarantee': True    # 防止孤立节点
}
```

### 何时调整参数

**调高 `predictor_threshold` (0.5 → 0.7)**:
- 需要更高质量的边
- 降低假阳性率
- 适用于高置信度预测场景

**调高 `top_k` (5 → 10)**:
- 增加图连通性
- 适用于稀疏数据
- 确保每个节点有足够邻居

**调低 `sim_threshold` (0.6 → 0.4)**:
- 增加边的多样性
- 适用于特征空间分散的数据

## 🚀 运行示例

```bash
# 使用PPI边预测器运行主pipeline
CUDA_VISIBLE_DEVICES=6 python robust_pipeline_edge.py
```

**预期输出**:
```
🛡️ 使用边预测器的鲁棒管道启动
🎯 鲁棒训练配置:
  - 目标比例: 90.0%
  - 使用边预测器: True
  - 边预测阈值: 0.5
  - Top-K保证: 5

🔗 阶段2.5: 初始化边预测器...
✅ 加载预训练边预测器: /path/to/models/edge_predictor_best.pth
   模型配置: input_dim=1280, hidden_dim=1024

🛡️ 阶段3: 使用边预测器的鲁棒增强...
✅ 增强完成: xxx 正样本, xxx 负样本
```

## ⚠️ 注意事项

1. **向后兼容性**:
   - `knn_k` 参数保留但不使用
   - 旧代码可继续运行，但建议更新

2. **性能要求**:
   - 需要加载PPI模型 (~50MB)
   - 边预测计算量略高于KNN

3. **模型依赖**:
   - 确保 `models/edge_predictor_best.pth` 存在
   - 如果缺失会回退到随机初始化（不推荐）

## 📊 验证

运行以下命令验证设置:
```bash
python verify_setup.py
```

应该看到:
```
✅ 训练好的PPI模型
   路径: /path/to/models/edge_predictor_best.pth
   模型配置: hidden_dim=1024
```

## 🔮 未来改进

1. **多物种支持**: 训练其他物种的PPI模型
2. **动态阈值**: 基于数据集自动调整阈值
3. **集成多源PPI**: 结合BioGRID、IntAct等数据库
4. **边权重**: 使用预测概率作为边权重

---

**更新日期**: 2025-10-20
**版本**: v2.0 - PPI边预测器版本
