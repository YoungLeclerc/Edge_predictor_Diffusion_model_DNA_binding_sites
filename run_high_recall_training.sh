#!/bin/bash
# 运行高召回率优化训练
# 用法: bash run_high_recall_training.sh

echo "================================================================================"
echo "🚀 DNA结合位点预测 - 高召回率优化训练"
echo "================================================================================"
echo ""
echo "优化策略:"
echo "  ✅ 1. 增加正样本权重 (Focal Alpha: 0.25→0.35, Pos Weight: 3.0x)"
echo "  ✅ 2. 降低困难样本关注 (Focal Gamma: 2.0→1.5)"
echo "  ✅ 3. 增加数据增强倍数 (5x→8x)"
echo "  ✅ 4. 放宽质量阈值 (0.5→0.4)"
echo "  ✅ 5. 降低Dropout (0.3→0.2)"
echo "  ✅ 6. 增加图连接 (Top-K: 5→8)"
echo "  ✅ 7. 增加训练轮数 (200→250 epochs)"
echo "  ✅ 8. 选择最高Recall模型"
echo ""
echo "预期效果:"
echo "  📈 Recall: +10-15%"
echo "  📈 F1 Score: +0~+5%"
echo "  📉 Precision: -3~-5% (可接受的tradeoff)"
echo ""
echo "================================================================================"
echo ""

# 检查必要文件
if [ ! -f "ultimate_pipeline_high_recall.py" ]; then
    echo "❌ 错误: ultimate_pipeline_high_recall.py 不存在"
    exit 1
fi

if [ ! -f "ultimate_config_high_recall.py" ]; then
    echo "❌ 错误: ultimate_config_high_recall.py 不存在"
    exit 1
fi

if [ ! -f "advanced_gnn_model.py" ]; then
    echo "❌ 错误: advanced_gnn_model.py 不存在"
    exit 1
fi

echo "✅ 所有必要文件已就绪"
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

echo ""
echo "================================================================================"
echo "开始训练..."
echo "================================================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行高召回率训练
python ultimate_pipeline_high_recall.py 2>&1 | tee high_recall_training.log

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================================================"
echo "✅ 训练完成!"
echo "================================================================================"
echo ""
echo "⏱️  总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📁 输出文件:"
echo "  • 训练日志: high_recall_training.log"
echo "  • 模型目录: Augmented_data_balanced/*_ultimate_high_recall_r050/"
echo "  • 结果文件: ultimate_high_recall_results.json"
echo "  • 模型权重: ultimate_gnn_model_high_recall.pt"
echo ""
echo "📊 查看结果:"
echo "  cat Augmented_data_balanced/DNA-573_Train_ultimate_high_recall_r050/ultimate_high_recall_results.json | jq '.test_results'"
echo ""
echo "💡 下一步:"
echo "  1. 对比标准版和高召回率版的性能"
echo "  2. 如果Recall提升不够,可以调整参数(参考 HIGH_RECALL_OPTIMIZATION_GUIDE.md)"
echo "  3. 使用高召回率模型进行预测"
echo ""
echo "================================================================================"
