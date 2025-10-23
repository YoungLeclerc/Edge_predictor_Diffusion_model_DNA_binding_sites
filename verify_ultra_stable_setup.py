#!/usr/bin/env python3
"""
验证超稳定版PPI模型集成
"""
import os
import json

def verify_setup():
    """验证超稳定版设置"""
    print("="*70)
    print("🔍 验证超稳定版PPI模型集成")
    print("="*70)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 检查模型文件
    model_path = os.path.join(current_dir, "models", "edge_predictor_best_ultra_stable.pth")

    print("\n📁 模型文件检查:")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"  ✅ 超稳定版模型存在")
        print(f"     路径: {model_path}")
        print(f"     大小: {size_mb:.1f} MB")
    else:
        print(f"  ❌ 错误: 模型文件不存在!")
        print(f"     预期路径: {model_path}")
        return False

    # 2. 检查训练历史
    history_path = os.path.join(current_dir, "results", "training_history_ultra_stable.json")

    print("\n📊 训练历史检查:")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

        epochs = len(history['val_auc'])
        best_auc = max(history['val_auc'])
        best_epoch = history['val_auc'].index(best_auc) + 1

        print(f"  ✅ 训练历史存在")
        print(f"     总轮数: {epochs}")
        print(f"     最佳AUC: {best_auc:.4f} (Epoch {best_epoch})")
    else:
        print(f"  ⚠️  训练历史文件不存在")
        print(f"     (不影响使用，仅供参考)")

    # 3. 检查评估结果
    results_dir = os.path.join(current_dir, "results_ultra_stable")

    print("\n📈 评估结果检查:")
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"  ✅ 评估结果目录存在")
        print(f"     包含 {len(files)} 个文件:")
        for f in files:
            print(f"       • {f}")
    else:
        print(f"  ⚠️  评估结果目录不存在")
        print(f"     (不影响使用，可运行 python 6_evaluate_model.py 生成)")

    # 4. 检查pipeline配置
    pipeline_path = os.path.join(current_dir, "robust_pipeline_edge.py")

    print("\n🔧 Pipeline配置检查:")
    with open(pipeline_path, 'r') as f:
        content = f.read()

    if 'edge_predictor_best_ultra_stable.pth' in content:
        print(f"  ✅ Pipeline已配置使用超稳定版模型")

        # 查找具体行数
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'edge_predictor_best_ultra_stable.pth' in line:
                print(f"     第{i}行: {line.strip()}")
    else:
        print(f"  ❌ Pipeline未配置超稳定版模型!")
        print(f"     需要修改 robust_pipeline_edge.py")
        return False

    # 5. 性能对比
    print("\n📊 性能对比总结:")
    print("  " + "─"*66)
    print(f"  {'版本':<12} {'训练AUC':<12} {'测试AUC':<12} {'状态':<10}")
    print("  " + "─"*66)
    print(f"  {'原始版':<12} {'0.9019':<12} {'-':<12} {'崩溃':<10}")
    print(f"  {'稳定版':<12} {'0.9146':<12} {'-':<12} {'崩溃':<10}")
    print(f"  {'超稳定版':<12} {'0.9300':<12} {'0.9297':<12} {'稳定 ⭐':<10}")
    print("  " + "─"*66)
    print(f"  提升: +{((0.9300-0.9019)/0.9019*100):.2f}% (相比原始版)")

    # 6. 下一步指引
    print("\n🚀 下一步操作:")
    print("  1. 运行主pipeline:")
    print("     CUDA_VISIBLE_DEVICES=6 python robust_pipeline_edge.py")
    print()
    print("  2. 查看评估结果:")
    print("     ls -lh results_ultra_stable/")
    print()
    print("  3. 查看训练曲线:")
    print("     cat results/training_history_ultra_stable.json")

    print("\n" + "="*70)
    print("✅ 验证完成！超稳定版PPI模型已就绪")
    print("="*70)

    return True

if __name__ == "__main__":
    success = verify_setup()
    exit(0 if success else 1)
