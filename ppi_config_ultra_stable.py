# 方案2配置文件 - 超稳定训练版本

import os

# 数据源
PPI_SOURCE = "string_db"
PPI_MIN_SCORE = 0.7

# 路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PPI_RAW_DIR = os.path.join(DATA_DIR, "ppi_raw")
PPI_PROCESSED_DIR = os.path.join(DATA_DIR, "ppi_processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# 创建目录
os.makedirs(PPI_RAW_DIR, exist_ok=True)
os.makedirs(PPI_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 特征配置
FEATURE_DIM = 1280
HIDDEN_DIM = 1024

# 🔥 超稳定训练参数
TRAIN_EPOCHS = 100                # ⬆️ 增加总轮数，因为学习率更低
BATCH_SIZE = 4096                 # 保持大batch

# 🔑 关键改进：大幅降低学习率
LEARNING_RATE = 0.0002            # ⬇️⬇️ 从0.0005降到0.0002 (降低60%)
WEIGHT_DECAY = 5e-4               # ⬆️ 更强的L2正则化 (从1e-4到5e-4)
WARMUP_EPOCHS = 5                 # ⬆️ 更长的warmup (从3到5)

# 学习率调度策略 - 使用ReduceLROnPlateau替代Cosine
LR_SCHEDULER = "plateau"          # 🔄 改用plateau策略，更稳定
LR_PATIENCE = 3                   # Plateau patience
LR_FACTOR = 0.5                   # 学习率衰减因子
LR_MIN = 1e-7                     # 最小学习率

# 数据参数
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
POS_NEG_RATIO = 1.0
MAX_EDGES = 50000

# A100优化选项
USE_AMP = True
GRADIENT_ACCUMULATION = 2         # ⬆️ 增加梯度累积 (模拟更大batch)
NUM_WORKERS = 16
GRADIENT_CLIP_NORM = 0.3          # ⬇️ 更严格的梯度裁剪 (从0.5到0.3)

# Early Stopping - 更保守
EARLY_STOPPING_PATIENCE = 15      # ⬆️ 增加patience (从10到15)
MIN_DELTA = 0.0001                # 最小改进阈值

# 设备配置
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 使用物理GPU 6
DEVICE = "cuda:0"  # 映射为逻辑GPU 0

# 日志
VERBOSE = True
LOG_INTERVAL = 1

# 🔧 额外的稳定性措施
USE_LABEL_SMOOTHING = 0.1         # 🆕 标签平滑，防止过拟合
DROPOUT_RATE = 0.1                # 🆕 在模型中添加dropout

print("""
🔗 方案2配置 - 超稳定训练版
────────────────────────────────
PPI来源: {}
特征维度: {}
训练轮数: {}
批处理大小: {}
学习率: {} (超稳定版 - 降低60%)
Warmup: {} epochs
梯度裁剪: {}
梯度累积: {} steps
LR调度: {}
设备: {}

🛡️ 稳定性改进:
  • 学习率降低60% (0.0005 → 0.0002)
  • 更强L2正则 (1e-4 → 5e-4)
  • 更长Warmup (3 → 5 epochs)
  • 更严格梯度裁剪 (0.5 → 0.3)
  • 梯度累积 (1 → 2 steps)
  • Plateau调度替代Cosine
  • 标签平滑 (0.1)
""".format(
    PPI_SOURCE, FEATURE_DIM, TRAIN_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, WARMUP_EPOCHS, GRADIENT_CLIP_NORM,
    GRADIENT_ACCUMULATION, LR_SCHEDULER, DEVICE
))
