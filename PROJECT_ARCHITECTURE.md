# 项目架构文档 - DNA结合位点预测系统

## 🏗️ 整体架构

### 系统流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    DNA Binding Site Prediction System                      │
│                 基于PPI引导的图神经网络预测系统                               │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                        │
   [数据准备阶段]                                          [模型训练测试阶段]
  (Steps 1-4)                                             (Steps 5-6 + Ultimate)
        │                                                        │
        │                                                        │
  ┌─────┴─────┐                                         ┌────────┴────────┐
  │           │                                         │                  │
  ↓           ↓                                         ↓                  ↓
┌────┐    ┌────────┐                            ┌──────────────┐   ┌────────────┐
│PPI │    │Protein │                            │Edge Predictor│   │   GNN      │
│Data│    │Sequence│                            │   Training   │   │  Training  │
└─┬──┘    └───┬────┘                            └──────┬───────┘   └─────┬──────┘
  │           │                                         │                 │
  │     ┌─────┴─────┐                                  │                 │
  │     │           │                                   │                 │
  │     ↓           ↓                                   │                 │
  │  ┌────────┐ ┌────────┐                            │                 │
  │  │ ESM2   │ │  PPI   │                            │                 │
  │  │Features│ │Process │                            │                 │
  │  └───┬────┘ └───┬────┘                            │                 │
  │      │          │                                   │                 │
  └──────┴──────────┴─────────────────────────────────┤                 │
                                                       │                 │
                                              ┌────────▼─────────────────▼───────┐
                                              │      ULTIMATE PIPELINE           │
                                              │   (Main Training-Test Script)    │
                                              └───────────────┬──────────────────┘
                                                              │
                                                              ↓
                                                      [Final Predictions]
```

---

## 📊 详细流程架构

### 阶段A: 数据准备（步骤1-4）

```
Step 1: Download PPI Data
┌─────────────────────────────────────────────┐
│   1_download_real_human_ppi.py              │
│                                             │
│   STRING Database (v12.0)                  │
│   └─> Homo sapiens (9606)                  │
│        └─> Confidence >= 400               │
│             └─> ~1,858,944 interactions    │
│                  └─> ~19,000 proteins      │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
              data/ppi_raw/
              ├── ppi_data.csv              # 过滤后的PPI数据
              └── protein_info.csv          # 蛋白质元数据
                   │
                   ↓
Step 2: Download Sequences
┌─────────────────────────────────────────────┐
│   2_download_protein_sequences.py          │
│                                             │
│   UniProt Database                         │
│   └─> 根据PPI网络中的蛋白质ID              │
│        └─> 下载FASTA序列                   │
│             └─> ~19,000 sequences          │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
              data/sequences/
              └── protein_sequences.fasta   # 所有蛋白质序列
                   │
                   ↓
Step 3: Extract ESM2 Features
┌─────────────────────────────────────────────┐
│   3_extract_esm2_features.py (GPU)         │
│                                             │
│   ESM2-t33-650M Model                      │
│   ├─> Input: FASTA序列                     │
│   ├─> Processing: Transformer              │
│   └─> Output: 1280-dim embeddings         │
│                                             │
│   Performance:                             │
│   ├─> GPU: A100 (40GB)                    │
│   ├─> Time: ~6-8 hours                    │
│   └─> Memory: ~30GB                       │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
              data/esm2_features/
              ├── ENSP00000000233.pt         # 每个蛋白质一个文件
              ├── ENSP00000000412.pt
              └── ... (~19,000 files)
                   │
                   ↓
Step 4: Preprocess PPI Data
┌─────────────────────────────────────────────┐
│   4_preprocess_ppi_data.py                 │
│                                             │
│   合并数据:                                 │
│   ├─> PPI网络结构                          │
│   ├─> 蛋白质序列                           │
│   └─> ESM2特征向量                         │
│                                             │
│   创建训练数据:                             │
│   ├─> 正样本: 有PPI相互作用 (1,858,944)   │
│   └─> 负样本: 无PPI相互作用 (随机采样)    │
│                                             │
│   数据格式:                                 │
│   └─> PyTorch Geometric格式               │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
              data/ppi_preprocessed/
              ├── train_data.pt              # 训练集
              ├── val_data.pt                # 验证集
              └── test_data.pt               # 测试集
```

### 阶段B: 边预测器训练（步骤5-6）

```
Step 5: Train Edge Predictor
┌─────────────────────────────────────────────┐
│   5_train_edge_predictor_ultra_stable.py   │
│                                             │
│   Model Architecture:                      │
│   ┌──────────────────────────────────────┐ │
│   │ ESM2 Features (1280-dim)             │ │
│   │         ↓                            │ │
│   │ Transform: 1280 → 1024               │ │
│   │    + BatchNorm + Dropout(0.3)       │ │
│   │         ↓                            │ │
│   │ Hidden1: 1024 → 512                  │ │
│   │    + ReLU + Dropout(0.3)            │ │
│   │         ↓                            │ │
│   │ Hidden2: 512 → 256                   │ │
│   │    + ReLU + Dropout(0.2)            │ │
│   │         ↓                            │ │
│   │ Output: 256 → 1 → Sigmoid            │ │
│   └──────────────────────────────────────┘ │
│                                             │
│   Training Config:                         │
│   ├─> Epochs: 50                           │
│   ├─> Batch Size: 2048                     │
│   ├─> Loss: BCEWithLogitsLoss              │
│   ├─> Optimizer: AdamW(lr=1e-3)            │
│   └─> Scheduler: CosineAnnealingLR         │
│                                             │
│   Performance:                              │
│   ├─> Training AUC: 0.930                  │
│   ├─> Validation AUC: 0.900                │
│   └─> Test AUC: 0.880                      │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
              models/
              └── edge_predictor_best_ultra_stable.pth
                   │
                   ↓
Step 6: Evaluate Edge Predictor
┌─────────────────────────────────────────────┐
│   6_evaluate_model.py                      │
│                                             │
│   Evaluation Metrics:                      │
│   ├─> AUC-ROC                              │
│   ├─> AUC-PR                               │
│   ├─> F1 Score                             │
│   ├─> Precision / Recall                   │
│   └─> Confusion Matrix                     │
│                                             │
│   Visualizations:                          │
│   ├─> ROC Curve                            │
│   ├─> PR Curve                             │
│   └─> Feature Importance                   │
└─────────────────────────────────────────────┘
```

### 阶段C: ULTIMATE Pipeline（主流程）

```
ULTIMATE Pipeline Architecture
┌───────────────────────────────────────────────────────────────────┐
│                 ultimate_pipeline.py                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Stage 1: Load Training Data                                 │ │
│  │   Input: DNA-573_Train.txt / DNA-646_Train.txt             │ │
│  │   Output: PyG Dataset                                       │ │
│  │   Stats: ~573/646 proteins                                  │ │
│  │          Original ratio: ~9% positive                       │ │
│  └─────────────────┬───────────────────────────────────────────┘ │
│                    │                                              │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ Stage 2: Train Enhanced Diffusion Model                     │ │
│  │                                                              │ │
│  │   ┌────────────────────────────────────────────────┐        │ │
│  │   │ EnhancedConditionalDiffusionModel              │        │ │
│  │   │                                                │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Context Encoder       │                     │        │ │
│  │   │ │ ├─> Multi-head Attn   │                     │        │ │
│  │   │ │ └─> Global Context    │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓ (Conditioning)                       │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Diffusion Process     │                     │        │ │
│  │   │ │ T=200 steps           │                     │        │ │
│  │   │ │ Forward + Reverse     │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓                                      │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Quality Evaluation    │                     │        │ │
│  │   │ │ ├─> Distribution Sim  │                     │        │ │
│  │   │ │ ├─> Validity Check    │                     │        │ │
│  │   │ │ └─> Range Check       │                     │        │ │
│  │   │ └───────────────────────┘                     │        │ │
│  │   └────────────────────────────────────────────────┘        │ │
│  │                                                              │ │
│  │   Config:                                                   │ │
│  │   ├─> T: 200 steps                                          │ │
│  │   ├─> Context dim: 256                                      │ │
│  │   ├─> Sample multiplier: 5x                                 │ │
│  │   └─> Quality threshold: 0.5                                │ │
│  │                                                              │ │
│  │   Output:                                                   │ │
│  │   └─> Trained diffusion model                               │ │
│  └─────────────────┬───────────────────────────────────────────┘ │
│                    │                                              │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ Stage 3: Load Edge Predictor                                │ │
│  │   Model: edge_predictor_best_ultra_stable.pth              │ │
│  │   Training AUC: 0.93                                        │ │
│  └─────────────────┬───────────────────────────────────────────┘ │
│                    │                                              │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ Stage 4: ULTIMATE Data Augmentation                         │ │
│  │                                                              │ │
│  │   For each protein:                                         │ │
│  │   ┌──────────────────────────────────────┐                 │ │
│  │   │ 1. Calculate target samples          │                 │ │
│  │   │    n_target = (total * ratio - pos)  │                 │ │
│  │   │                                       │                 │ │
│  │   │ 2. Generate 5x candidates            │                 │ │
│  │   │    ├─> Diffusion model generates     │                 │ │
│  │   │    └─> Quality scores computed       │                 │ │
│  │   │                                       │                 │ │
│  │   │ 3. Filter by quality                 │                 │ │
│  │   │    ├─> Threshold: 0.5                │                 │ │
│  │   │    └─> Select top-k                  │                 │ │
│  │   │                                       │                 │ │
│  │   │ 4. Build graph with Edge Predictor   │                 │ │
│  │   │    ├─> Predict PPI scores            │                 │ │
│  │   │    ├─> Create edges                  │                 │ │
│  │   │    └─> Add edge features             │                 │ │
│  │   └──────────────────────────────────────┘                 │ │
│  │                                                              │ │
│  │   Output Statistics:                                        │ │
│  │   ├─> Generated samples: ~65K-130K                          │ │
│  │   ├─> Average quality: 0.45-0.60                            │ │
│  │   ├─> Diversity: 0.99-1.00                                  │ │
│  │   └─> Final ratio: ~50%                                     │ │
│  │                                                              │ │
│  │   Time: ~50-60 min (573 proteins, A100)                    │ │
│  └─────────────────┬───────────────────────────────────────────┘ │
│                    │                                              │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ Stage 5: Cross-Validation Training                          │ │
│  │                                                              │ │
│  │   ┌────────────────────────────────────────────────┐        │ │
│  │   │ AdvancedBindingSiteGNN                         │        │ │
│  │   │                                                │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Input Projection      │                     │        │ │
│  │   │ │ 1280 → 256            │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓                                      │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Multi-Scale GAT (×4)  │                     │        │ │
│  │   │ │ ├─> GATv2Conv         │                     │        │ │
│  │   │ │ ├─> 4 heads           │                     │        │ │
│  │   │ │ ├─> LayerNorm         │                     │        │ │
│  │   │ │ └─> Residual          │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓                                      │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Global Pooling        │                     │        │ │
│  │   │ │ Mean + Max            │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓                                      │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Local-Global Fusion   │                     │        │ │
│  │   │ └───────┬───────────────┘                     │        │ │
│  │   │         │                                      │        │ │
│  │   │         ↓                                      │        │ │
│  │   │ ┌───────────────────────┐                     │        │ │
│  │   │ │ Classifier            │                     │        │ │
│  │   │ │ 512 → 1               │                     │        │ │
│  │   │ └───────────────────────┘                     │        │ │
│  │   └────────────────────────────────────────────────┘        │ │
│  │                                                              │ │
│  │   Loss Function:                                            │ │
│  │   ├─> Focal Loss (α=0.25, γ=2.0)                           │ │
│  │   └─> Class Balanced Weight                                 │ │
│  │                                                              │ │
│  │   Training Strategy:                                        │ │
│  │   ├─> 3-Fold Cross-Validation                               │ │
│  │   ├─> Early Stopping (patience=15)                          │ │
│  │   └─> Best model selection                                  │ │
│  │                                                              │ │
│  │   Time: ~15-20 min                                          │ │
│  └─────────────────┬───────────────────────────────────────────┘ │
│                    │                                              │
│  ┌─────────────────▼───────────────────────────────────────────┐ │
│  │ Stage 6: Test on Multiple Datasets                          │ │
│  │                                                              │ │
│  │   Test Sets:                                                │ │
│  │   ├─> DNA-573_Test.txt                                      │ │
│  │   ├─> DNA-646_Test.txt                                      │ │
│  │   └─> DNA-181_Test.txt                                      │ │
│  │                                                              │ │
│  │   Metrics Computed:                                         │ │
│  │   ├─> F1 Score                                              │ │
│  │   ├─> MCC (Matthews Correlation Coefficient)                │ │
│  │   ├─> Accuracy                                              │ │
│  │   ├─> AUC-PR (Area Under Precision-Recall Curve)           │ │
│  │   └─> AUC-ROC (Area Under ROC Curve)                        │ │
│  │                                                              │ │
│  │   Time: ~5-10 min                                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│   Total Pipeline Time: ~70-90 minutes (DNA-573, A100)           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🔬 核心组件详解

### 1. 增强版条件扩散模型

```
EnhancedConditionalDiffusionModel
├── Context Encoder (提取全局特征)
│   ├─> Input: Protein features (all nodes)
│   ├─> Multi-head Attention (4 heads)
│   ├─> Output: Global context (256-dim)
│   └─> Used for: Conditioning diffusion process
│
├── Diffusion Scheduler
│   ├─> Noise Schedule: Linear (β₁=1e-4 to βT=0.02)
│   ├─> Steps: T=200
│   └─> Complexity-aware: Adjust based on protein complexity
│
├── Denoiser Network
│   ├─> Input: [x_t, global_context, time_embedding]
│   ├─> Architecture: MLP(1280+256+1 → 512 → 512 → 1280)
│   ├─> Activation: SiLU
│   └─> Normalization: LayerNorm
│
└── Quality Evaluation
    ├─> Distribution Similarity: exp(-dist/5.0)  [60%]
    ├─> Validity Check: exp(-max_norm/3.0)      [25%]
    └─> Range Check: exp(-penalty/5.0)          [15%]
```

### 2. 高级GAT-GNN模型

```
AdvancedBindingSiteGNN
├── Input Projection
│   └─> Linear(1280 → 256)
│
├── Multi-Scale GAT Layers (×4)
│   ├─> Layer 1: GATv2Conv(256, 64, heads=4) → 256
│   ├─> Layer 2: GATv2Conv(256, 64, heads=4) → 256
│   ├─> Layer 3: GATv2Conv(256, 64, heads=4) → 256
│   └─> Layer 4: GATv2Conv(256, 64, heads=4) → 256
│
│   Each layer includes:
│   ├─> GATv2Conv (attention-based message passing)
│   ├─> LayerNorm (stabilize training)
│   ├─> Residual Connection (skip connection)
│   └─> Dropout(0.3) (regularization)
│
├── Global Pooling
│   ├─> Global Mean Pool
│   ├─> Global Max Pool
│   └─> Concatenate → 512-dim
│
├── Local-Global Fusion
│   ├─> Broadcast global info to nodes
│   ├─> Concatenate [local, global] → 512-dim
│   └─> Fusion MLP → 512-dim
│
└── Classifier
    └─> Linear(512 → 1) + Sigmoid
```

### 3. PPI边预测器

```
ImprovedEdgePredictor
├── Transform Layer
│   ├─> Linear(1280 → 1024)
│   ├─> BatchNorm1d
│   ├─> ReLU
│   └─> Dropout(0.3)
│
├── Hidden Layer 1
│   ├─> Linear(1024 → 512)
│   ├─> ReLU
│   └─> Dropout(0.3)
│
├── Hidden Layer 2
│   ├─> Linear(512 → 256)
│   ├─> ReLU
│   └─> Dropout(0.2)
│
└── Output Layer
    ├─> Linear(256 → 1)
    └─> Sigmoid

Training:
├─> Data: STRING PPI (1.86M interactions)
├─> Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
├─> Scheduler: CosineAnnealingLR
├─> Loss: BCEWithLogitsLoss
└─> Performance: AUC=0.93 (training), 0.88 (test)
```

---

## 📁 数据流

### 输入数据

```
Raw Datasets:
├── DNA-573_Train.txt           # 573个蛋白质，~14K正样本，~145K负样本
├── DNA-573_Test.txt
├── DNA-646_Train.txt           # 646个蛋白质，~16K正样本，~298K负样本
├── DNA-646_Test.txt
└── DNA-181_Test.txt

PPI Data (from STRING):
├── ppi_data.csv                # 1.86M protein interactions
└── protein_info.csv            # 19K protein metadata

Protein Sequences:
└── protein_sequences.fasta     # 19K protein FASTA sequences

ESM2 Features:
└── esm2_features/              # 19K × 1280-dim feature vectors
    ├── ENSP00000000233.pt
    └── ...
```

### 中间数据

```
After Augmentation:
├── Augmented proteins          # Original + Generated nodes
│   ├─> Original nodes: ~14K-16K (positive)
│   └─> Generated nodes: ~65K-130K (positive)
│
├── Graph Structure
│   ├─> Nodes: ~159K-314K total
│   ├─> Edges: ~1.7M-2.4M (from Edge Predictor)
│   └─> Edge Features: PPI confidence scores
│
└── Data Statistics
    ├─> Positive ratio: ~50% (from ~9%)
    ├─> Average quality: 0.45-0.60
    └─> Diversity: 0.99-1.00
```

### 输出结果

```
Models:
├── models/edge_predictor_best_ultra_stable.pth
└── Augmented_data_balanced/
    └── DNA-573_Train_ultimate_r50/
        └── ultimate_gnn_model.pt

Results:
└── Augmented_data_balanced/
    ├── DNA-573_Train_ultimate_r50/
    │   └── ultimate_results.json
    │       ├─> training_info
    │       ├─> test_results
    │       └─> augmentation_stats
    └── ultimate_pipeline_summary.json
```

---

## ⚙️ 关键参数配置

### ultimate_config.py

```python
# 数据增强
target_ratio = 0.5                    # 目标正样本比例（50%）
quality_threshold = 0.5               # 质量阈值
min_samples_per_protein = 5           # 每个蛋白质最少生成样本数

# 增强版扩散模型
use_enhanced_diffusion = True
enhanced_diffusion_config = {
    'T': 200,                         # 扩散步数
    'hidden_dim': 512,                # 隐藏层维度
    'context_dim': 256,               # 上下文维度
    'max_attempts': 3,                # 采样尝试次数
    'sample_multiplier': 5,           # 候选样本倍数（生成5x，选top-1x）
}

# 高级GNN
use_advanced_gnn = True
advanced_gnn_config = {
    'hidden_dim': 256,                # 隐藏层维度
    'num_layers': 4,                  # GAT层数
    'heads': 4,                       # 注意力头数
    'dropout': 0.3,                   # Dropout率
    'use_edge_features': True,        # 使用边特征
    'focal_alpha': 0.25,              # Focal loss α
    'focal_gamma': 2.0,               # Focal loss γ
    'class_balanced': True,           # 类别平衡损失
}

# 边预测器
use_edge_predictor = True
edge_predictor_config = {
    'predictor_threshold': 0.8,       # PPI预测阈值
    'sim_threshold': 0.7,             # 相似度阈值
    'dist_threshold': 1.2,            # 距离阈值
    'top_k': 5,                       # Top-K邻居
}

# 训练
cv_folds = 3                          # 交叉验证折数
diffusion_epochs = 100                # 扩散模型训练轮数
gnn_epochs = 200                      # GNN训练轮数
gnn_patience = 15                     # Early stopping patience
seed = 42                             # 随机种子
```

---

## 🔧 性能优化

### GPU加速优化

```
优化项目:
├── 保持数据在GPU上
│   └─> 避免CPU-GPU数据传输
│
├── 批量操作
│   ├─> 批量生成样本（不循环）
│   ├─> GPU topk选择
│   └─> GPU张量操作
│
├── Inference模式
│   └─> torch.inference_mode() (更快than no_grad)
│
├── 时间嵌入缓存
│   └─> 缓存重复计算的时间步嵌入
│
└── Fused操作
    └─> 预计算常量，减少kernel启动

性能提升:
├─> 数据增强: 18s/it → 5s/it (3.6x)
├─> GPU利用率: 30% → 75% (2.5x)
└─> 总时间: 7小时 → 1.5小时 (4.7x)
```

### 内存优化

```
优化策略:
├── Batch size动态调整
├── 梯度累积
├── 混合精度训练 (可选)
└── 清理中间变量

内存使用:
├─> Edge Predictor Training: ~25GB
├─> Diffusion Model Training: ~15GB
├─> GNN Training: ~10GB
└─> Total Peak: ~30GB (A100 40GB OK)
```

---

## 📊 性能基准

### 时间消耗（DNA-573, A100 40GB）

```
阶段                         时间         GPU利用率
────────────────────────────────────────────────
1. 加载数据                  <1min        0%
2. 训练扩散模型              ~4min        80-90%
3. 加载边预测器              <1min        0%
4. 数据增强 (573蛋白)        ~50min       70-85%
5. 交叉验证训练GNN            ~15min       85-95%
6. 测试集评估                ~5min        60-70%
────────────────────────────────────────────────
总计                         ~75min       -
```

### 模型性能

```
数据集          F1      MCC     AUC-PR   AUC-ROC
─────────────────────────────────────────────────
DNA-573        0.583   0.552   0.648    0.826
DNA-646        0.491   0.461   0.568    0.803
DNA-181        0.512   0.485   0.591    0.815
```

---

## 🎯 For Graphical Abstract

### 推荐可视化元素

1. **输入层**: DNA结合位点数据（不平衡，9%正样本）
2. **PPI模块**: STRING数据库 → 边预测器（AUC=0.93）
3. **扩散模块**: 条件扩散模型 → 高质量样本生成
4. **GNN模块**: Multi-scale GAT → 最终预测
5. **输出层**: 性能提升指标（F1 +28-60%）

### 配色建议

- 输入/输出: 蓝色系 (#2196F3)
- PPI/边预测器: 绿色系 (#4CAF50)
- 扩散模型: 紫色系 (#9C27B0)
- GNN模型: 橙色系 (#FF9800)
- 性能指标: 红色系 (#F44336)

---

**文档版本**: v1.0
**最后更新**: 2024-10-22
**作者**: Claude + User
