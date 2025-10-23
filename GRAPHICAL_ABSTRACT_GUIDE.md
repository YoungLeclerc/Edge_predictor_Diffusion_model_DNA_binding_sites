# Graphical Abstract 设计指南

## 🎨 推荐的图形摘要设计

### 方案A：横向流程图（推荐）

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  输入数据              核心技术创新                            输出结果      │
│  ────────            ─────────────────────────               ────────     │
│                                                                            │
│  ┌─────────┐         ┌──────────────────────┐              ┌───────────┐ │
│  │DNA结合   │         │ ① PPI边预测器        │              │ 预测性能  │ │
│  │位点数据  │  ─────> │  STRING数据库        │  ──────┐     │           │ │
│  │         │         │  • 1.86M interactions│        │     │ F1:+28%   │ │
│  │9%正样本 │         │  • AUC=0.93          │        │     │ MCC:+30%  │ │
│  │(不平衡) │         └──────────────────────┘        │     │ AUC:+28%  │ │
│  └─────────┘                                         ↓     │           │ │
│                      ┌──────────────────────┐   ┌────────┐ │           │ │
│                      │ ② 增强版扩散模型      │   │ 图结构  │ │           │ │
│                      │  Context-Guided       │   │ 构建   │ └───────────┘ │
│                      │  • 质量: 0.45-0.60   │   └────┬───┘               │
│                      │  • 多样性: 0.99      │        │                   │
│                      └──────────┬───────────┘        │                   │
│                                 │                    │                   │
│                                 ↓ (50%平衡数据)      │                   │
│                      ┌──────────────────────┐        │                   │
│                      │ ③ 高级GAT-GNN        │ <──────┘                   │
│                      │  • Multi-head Attn   │                            │
│                      │  • Focal Loss        │                            │
│                      │  • Class Balance     │                            │
│                      └──────────────────────┘                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

标注:
• STRING数据库图标
• 扩散过程动画效果
• GAT注意力可视化
• 性能柱状图对比
```

### 方案B：垂直流程图

```
┌──────────────────────────────────────┐
│        DNA Binding Site               │
│       Prediction System               │
│   基于PPI引导的图神经网络               │
└──────────────────┬───────────────────┘
                   │
       ┌───────────┴──────────┐
       │                      │
   [输入数据]            [已知信息]
       │                      │
       ↓                      ↓
┌─────────────┐      ┌─────────────────┐
│DNA结合位点  │      │ STRING PPI数据  │
│数据集       │      │ 1.86M interactions│
│• 573蛋白质  │      │ ~19K proteins    │
│• 9%正样本   │      └────────┬─────────┘
│  (不平衡)   │               │
└──────┬──────┘               │
       │                      │
       └──────────┬───────────┘
                  │
         ┌────────▼────────┐
         │  创新技术模块    │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ↓             ↓             ↓
┌────────┐  ┌──────────┐  ┌──────────┐
│ PPI    │  │ Enhanced │  │ Advanced │
│ Edge   │  │ Diffusion│  │ GAT-GNN  │
│Predictor│  │  Model   │  │  Model   │
│        │  │          │  │          │
│AUC=0.93│  │Quality   │  │Multi-head│
│        │  │0.45-0.60 │  │Attention │
└───┬────┘  └────┬─────┘  └─────┬────┘
    │            │              │
    │     ┌──────▼─────────┐   │
    │     │ Data Augment   │   │
    │     │ 9% → 50%      │   │
    │     │ Diversity:0.99 │   │
    │     └───────┬────────┘   │
    │            │             │
    └────────────┴─────────────┘
                 │
         ┌───────▼────────┐
         │   Final Model   │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Performance    │
         │  • F1: +28-60% │
         │  • MCC: +30-65%│
         │  • AUC: +28-59%│
         └─────────────────┘
```

---

## 🎨 配色方案

### 主色调（推荐）

```
输入数据:
  - 主色: #2196F3 (蓝色)
  - 辅色: #BBDEFB (浅蓝)

PPI边预测器:
  - 主色: #4CAF50 (绿色)
  - 辅色: #C8E6C9 (浅绿)
  - 强调色: #1B5E20 (深绿) [用于高性能标注]

扩散模型:
  - 主色: #9C27B0 (紫色)
  - 辅色: #E1BEE7 (浅紫)
  - 强调色: #4A148C (深紫) [用于质量指标]

GNN模型:
  - 主色: #FF9800 (橙色)
  - 辅色: #FFE0B2 (浅橙)
  - 强调色: #E65100 (深橙) [用于结构标注]

输出结果:
  - 主色: #F44336 (红色)
  - 辅色: #FFCDD2 (浅红)
  - 强调色: #B71C1C (深红) [用于性能提升]

连接线:
  - 主流程: #424242 (深灰)
  - 数据流: #757575 (中灰)
  - 辅助线: #BDBDBD (浅灰)
```

### 替代配色（学术风格）

```
输入: #1E88E5 (专业蓝)
PPI: #43A047 (自然绿)
扩散: #8E24AA (科学紫)
GNN: #FB8C00 (温暖橙)
输出: #E53935 (强调红)
```

---

## 📐 布局建议

### 版式A：横向布局 (16:9)

```
宽度: 1600px
高度: 900px

布局:
├─ 标题区 (顶部): 200px
│  ├─ 标题: "DNA Binding Site Prediction"
│  └─ 副标题: "PPI-Guided Graph Neural Networks"
│
├─ 主体区 (中部): 600px
│  ├─ 左侧 (输入): 300px
│  ├─ 中部 (创新): 800px
│  └─ 右侧 (输出): 300px
│
└─ 性能区 (底部): 100px
   └─ 柱状图/指标对比
```

### 版式B：方形布局 (1:1)

```
尺寸: 1200px × 1200px

布局:
├─ 标题区 (顶部): 150px
├─ 输入区: 200px
├─ 创新技术区: 600px
│  ├─ PPI模块: 200px
│  ├─ 扩散模块: 200px
│  └─ GNN模块: 200px
├─ 输出区: 150px
└─ 指标区: 100px
```

---

## 🖼️ 关键视觉元素

### 1. 输入数据可视化

```
图标: 📊 不平衡条形图
显示: 正样本9% (红色) vs 负样本91% (灰色)

或

饼图:
- 正样本: 9% (小块，红色)
- 负样本: 91% (大块，灰色)

文字标注:
"DNA-573: 14,479 positive / 145,404 negative"
"Highly Imbalanced (1:10)"
```

### 2. PPI边预测器

```
图标: 🌐 蛋白质互作网络图
显示: 节点(蛋白质) + 边(相互作用)

标注:
"STRING Database"
"1.86M Interactions"
"Training AUC: 0.93"

可选: 小图展示边预测器架构
```

### 3. 扩散模型

```
图标: 🎨 前向-反向扩散过程动画
显示: x₀ → x₁ → ... → xₜ → ... → x₀'

标注:
"Conditional Diffusion"
"Context-Guided Generation"
"Quality: 0.45-0.60"
"Diversity: 0.99"

可选: 质量分布直方图
```

### 4. GAT-GNN模型

```
图标: 🧠 多层神经网络结构
显示: 输入层 → GAT层 × 4 → 输出层

标注:
"Multi-head Attention"
"4 Layers × 4 Heads"
"Focal Loss + Class Balance"

可选: 注意力权重热力图
```

### 5. 性能对比

```
柱状图:
X轴: 方法 (Baseline, +Diffusion, +Edge, ULTIMATE)
Y轴: 性能指标

三组柱子:
- F1 Score (蓝色)
- MCC (绿色)
- AUC-PR (橙色)

标注提升百分比:
F1: +28.1% ↑
MCC: +29.9% ↑
AUC: +27.6% ↑
```

---

## 📊 数据可视化建议

### 性能对比图（必须包含）

```python
# 使用matplotlib/seaborn生成

import matplotlib.pyplot as plt
import numpy as np

methods = ['Baseline', '+Diffusion', '+Edge', 'ULTIMATE']
f1_scores = [0.455, 0.480, 0.510, 0.583]
mcc_scores = [0.425, 0.448, 0.475, 0.552]
auc_scores = [0.508, 0.531, 0.562, 0.648]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, f1_scores, width, label='F1', color='#2196F3')
ax.bar(x, mcc_scores, width, label='MCC', color='#4CAF50')
ax.bar(x + width, auc_scores, width, label='AUC-PR', color='#FF9800')

ax.set_xlabel('Methods', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Performance Comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
```

### 数据增强效果图

```python
# 对比增强前后的正负样本比例

import matplotlib.pyplot as plt

categories = ['Original', 'After Augmentation']
positive = [9.06, 50.0]
negative = [90.94, 50.0]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(categories, positive, label='Positive', color='#F44336')
ax.barh(categories, negative, left=positive, label='Negative', color='#9E9E9E')

ax.set_xlabel('Percentage (%)', fontsize=14)
ax.set_title('Data Augmentation Effect', fontsize=16)
ax.legend()

plt.tight_layout()
plt.savefig('augmentation_effect.png', dpi=300)
```

---

## 🎯 文字标注建议

### 简洁版标注（推荐用于Abstract）

```
标题: "PPI-Guided GNN for DNA Binding Site Prediction"

输入: "Imbalanced Data (9% positive)"

核心技术:
① "PPI Edge Predictor (AUC=0.93)"
② "Enhanced Diffusion (Quality=0.50)"
③ "Advanced GAT-GNN"

输出: "Performance: F1 +28%, MCC +30%, AUC +28%"
```

### 详细版标注（用于Poster）

```
标题:
"DNA Binding Site Prediction with Protein-Protein
Interaction Guided Graph Neural Networks"

输入区:
"DNA Binding Site Dataset"
"• DNA-573: 573 proteins"
"• DNA-646: 646 proteins"
"• Highly Imbalanced: 9% positive samples"

技术模块:

① PPI Edge Predictor
"• Trained on STRING Database"
"• 1.86M Human PPI Interactions"
"• Model Performance: AUC=0.93"
"• Used to construct biologically meaningful graphs"

② Enhanced Conditional Diffusion Model
"• Context-guided sample generation"
"• T=200 diffusion steps"
"• Quality-aware sampling (5x candidates)"
"• Output Quality: 0.45-0.60, Diversity: 0.99"
"• Achieves 50% balanced ratio"

③ Advanced GAT-GNN Model
"• Multi-scale Graph Attention Networks"
"• 4 layers × 4-head attention"
"• Focal Loss (α=0.25, γ=2.0)"
"• Class Balanced Loss"

性能提升:
"DNA-573:"
"• F1 Score: 0.455 → 0.583 (+28.1%)"
"• MCC: 0.425 → 0.552 (+29.9%)"
"• AUC-PR: 0.508 → 0.648 (+27.6%)"

"DNA-646:"
"• F1 Score: 0.306 → 0.491 (+60.5%)"
"• MCC: 0.280 → 0.461 (+64.6%)"
"• AUC-PR: 0.358 → 0.568 (+58.7%)"
```

---

## 🖌️ 设计工具推荐

### 专业设计软件

1. **Adobe Illustrator** (推荐)
   - 矢量图形
   - 精确控制
   - 高分辨率输出

2. **Inkscape** (免费替代)
   - 开源矢量编辑器
   - 功能强大

3. **BioRender** (生物医学专用)
   - 预置生物图标
   - 快速创建
   - 付费服务

### 在线工具

1. **Figma** (推荐)
   - 在线协作
   - 模板丰富
   - 免费版可用

2. **Canva**
   - 简单易用
   - 模板多样

### Python绘图

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 创建画布
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# 标题
ax.text(8, 8.5, 'DNA Binding Site Prediction',
        ha='center', va='center', fontsize=24, weight='bold')

# 输入框
input_box = FancyBboxPatch((0.5, 3), 2, 2,
                           boxstyle="round,pad=0.1",
                           facecolor='#BBDEFB',
                           edgecolor='#2196F3', linewidth=2)
ax.add_patch(input_box)
ax.text(1.5, 4, 'Input Data\n9% Positive',
        ha='center', va='center', fontsize=12)

# ... 继续添加其他元素

plt.tight_layout()
plt.savefig('graphical_abstract.png', dpi=300, bbox_inches='tight')
```

---

## ✅ 检查清单

在提交Graphical Abstract前，确保：

- [ ] 清晰展示三大核心创新点
- [ ] 包含性能提升数据
- [ ] 配色专业协调
- [ ] 文字大小合适（可读性）
- [ ] 流程逻辑清晰
- [ ] 高分辨率输出（至少300 DPI）
- [ ] 符合期刊要求（尺寸、格式）
- [ ] 所有文字无拼写错误
- [ ] 图标和箭头对齐
- [ ] 包含必要的图例

---

## 📝 期刊特定要求

### Nature系列
- 尺寸: 宽度 83-170mm, 高度不限
- 格式: TIFF, EPS, PDF
- 分辨率: 300-600 DPI
- 配色: 清晰对比，避免过于鲜艳

### Science
- 尺寸: 宽度 5.5-11 cm
- 格式: PDF, EPS
- 分辨率: 300+ DPI
- 风格: 简洁明了

### Bioinformatics
- 尺寸: 宽度 < 180mm
- 格式: TIFF, PDF
- 分辨率: 300+ DPI
- 风格: 学术专业

---

## 💡 最终建议

1. **保持简洁**: 不要过度复杂化，重点突出三大创新
2. **数据说话**: 用具体数字展示性能提升
3. **视觉引导**: 使用箭头和流程线引导视线
4. **配色一致**: 整个图使用统一的配色方案
5. **高质量输出**: 确保矢量图形和高DPI
6. **多次审阅**: 请同事/导师审阅并提供反馈

---

**推荐组合**: 方案A横向流程 + 专业配色 + 性能对比图

**预计制作时间**: 2-4小时（使用Adobe Illustrator或BioRender）

**最终格式**: PDF (矢量) + PNG (300 DPI) 备份
