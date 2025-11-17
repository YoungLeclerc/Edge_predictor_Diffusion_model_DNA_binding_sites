#!/usr/bin/env python3
"""
对比标准版和高召回率版的性能
"""
import json
import os
import sys
from pathlib import Path


def load_results(result_file):
    """加载结果文件"""
    if not os.path.exists(result_file):
        return None

    with open(result_file, 'r') as f:
        return json.load(f)


def print_comparison(baseline_results, optimized_results):
    """打印对比结果"""

    print("=" * 100)
    print("📊 性能对比: 标准版 vs 高召回率优化版")
    print("=" * 100)
    print()

    # 对比每个测试集
    test_sets = set(baseline_results['test_results'].keys()) & set(optimized_results['test_results'].keys())

    for test_name in sorted(test_sets):
        baseline = baseline_results['test_results'][test_name]
        optimized = optimized_results['test_results'][test_name]

        print(f"\n🔬 测试集: {test_name}")
        print("-" * 100)
        print(f"{'指标':<15} {'标准版':<12} {'高召回率版':<12} {'变化':<15} {'说明':<30}")
        print("-" * 100)

        metrics = [
            ('Recall', 'recall', '⭐ 主要优化目标'),
            ('F1', 'f1', '综合指标'),
            ('Precision', 'precision', '可能略微下降'),
            ('Specificity', 'specificity', '特异性'),
            ('MCC', 'mcc', '马修斯相关系数'),
            ('Accuracy', 'accuracy', '准确率'),
            ('AUC-PR', 'auc_pr', 'PR曲线下面积'),
            ('AUC-ROC', 'auc_roc', 'ROC曲线下面积'),
        ]

        improvements = []

        for display_name, key, note in metrics:
            baseline_val = baseline.get(key, 0)
            optimized_val = optimized.get(key, 0)

            if baseline_val > 0:
                delta = optimized_val - baseline_val
                delta_pct = (delta / baseline_val) * 100

                # 格式化变化
                if delta > 0:
                    change_str = f"+{delta:.4f} ({delta_pct:+.1f}%)"
                    emoji = "📈"
                elif delta < 0:
                    change_str = f"{delta:.4f} ({delta_pct:.1f}%)"
                    emoji = "📉"
                else:
                    change_str = "0.0000 (0.0%)"
                    emoji = "➡️"

                print(f"{display_name:<15} {baseline_val:<12.4f} {optimized_val:<12.4f} {emoji} {change_str:<12} {note:<30}")

                improvements.append((display_name, delta, delta_pct))
            else:
                print(f"{display_name:<15} {baseline_val:<12.4f} {optimized_val:<12.4f} {'N/A':<15} {note:<30}")

        # 混淆矩阵对比
        if 'confusion_matrix' in baseline and 'confusion_matrix' in optimized:
            print()
            print(f"混淆矩阵对比:")

            baseline_cm = baseline['confusion_matrix']
            optimized_cm = optimized['confusion_matrix']

            tp_base = baseline_cm.get('TP', baseline_cm.get('tp', 0))
            fn_base = baseline_cm.get('FN', baseline_cm.get('fn', 0))
            fp_base = baseline_cm.get('FP', baseline_cm.get('fp', 0))
            tn_base = baseline_cm.get('TN', baseline_cm.get('tn', 0))

            tp_opt = optimized_cm.get('TP', optimized_cm.get('tp', 0))
            fn_opt = optimized_cm.get('FN', optimized_cm.get('fn', 0))
            fp_opt = optimized_cm.get('FP', optimized_cm.get('fp', 0))
            tn_opt = optimized_cm.get('TN', optimized_cm.get('tn', 0))

            print(f"  {'':>15} {'标准版':>20} {'高召回率版':>20} {'变化':>15}")
            print(f"  {'TP (真阳性)':>15} {tp_base:>20} {tp_opt:>20} {tp_opt - tp_base:>+15}")
            print(f"  {'FN (假阴性)':>15} {fn_base:>20} {fn_opt:>20} {fn_opt - fn_base:>+15} 📉 应该减少")
            print(f"  {'FP (假阳性)':>15} {fp_base:>20} {fp_opt:>20} {fp_opt - fp_base:>+15} 可能增加")
            print(f"  {'TN (真阴性)':>15} {tn_base:>20} {tn_opt:>20} {tn_opt - tn_base:>+15}")

        # 关键改进总结
        print()
        print("关键改进:")
        recall_improvement = [x for x in improvements if x[0] == 'Recall'][0]
        f1_improvement = [x for x in improvements if x[0] == 'F1'][0]
        precision_improvement = [x for x in improvements if x[0] == 'Precision'][0]

        print(f"  • Recall变化: {recall_improvement[1]:+.4f} ({recall_improvement[2]:+.1f}%)")
        print(f"  • F1变化: {f1_improvement[1]:+.4f} ({f1_improvement[2]:+.1f}%)")
        print(f"  • Precision变化: {precision_improvement[1]:+.4f} ({precision_improvement[2]:+.1f}%)")

        print()

    # 训练信息对比
    print("\n" + "=" * 100)
    print("🔧 训练配置对比")
    print("=" * 100)
    print()

    if 'training_info' in baseline_results and 'training_info' in optimized_results:
        baseline_info = baseline_results['training_info']
        optimized_info = optimized_results['training_info']

        print(f"{'配置项':<30} {'标准版':<20} {'高召回率版':<20}")
        print("-" * 70)

        if 'config_summary' in optimized_info:
            config = optimized_info['config_summary']
            print(f"{'Focal Alpha (正样本权重)':<30} {'0.25':<20} {config.get('focal_alpha', 'N/A'):<20}")
            print(f"{'Focal Gamma (困难样本关注)':<30} {'2.0':<20} {config.get('focal_gamma', 'N/A'):<20}")
            print(f"{'正样本额外权重':<30} {'1.0':<20} {config.get('pos_weight', 'N/A'):<20}")
            print(f"{'质量阈值':<30} {'0.8':<20} {config.get('quality_threshold', 'N/A'):<20}")
            print(f"{'采样倍数':<30} {'5':<20} {config.get('sample_multiplier', 'N/A'):<20}")

        print()
        print(f"{'数据增强统计':<30} {'标准版':<20} {'高召回率版':<20}")
        print("-" * 70)
        print(f"{'增强后正样本数':<30} {baseline_info.get('augmented_positive', 'N/A'):<20} {optimized_info.get('augmented_positive', 'N/A'):<20}")
        print(f"{'增强后负样本数':<30} {baseline_info.get('augmented_negative', 'N/A'):<20} {optimized_info.get('augmented_negative', 'N/A'):<20}")
        print(f"{'增强后比例':<30} {baseline_info.get('augmented_ratio', 0):<20.3f} {optimized_info.get('augmented_ratio', 0):<20.3f}")

    print()
    print("=" * 100)
    print("✅ 对比完成")
    print("=" * 100)


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python compare_recall_results.py <baseline_json> <optimized_json>")
        print()
        print("示例:")
        print("  python compare_recall_results.py \\")
        print("    Augmented_data_balanced/DNA-573_Train_ultimate_r050/ultimate_results.json \\")
        print("    Augmented_data_balanced/DNA-573_Train_ultimate_high_recall_r050/ultimate_high_recall_results.json")
        sys.exit(1)

    baseline_file = sys.argv[1]
    optimized_file = sys.argv[2]

    print(f"\n加载结果文件...")
    print(f"  标准版: {baseline_file}")
    print(f"  高召回率版: {optimized_file}")
    print()

    baseline_results = load_results(baseline_file)
    optimized_results = load_results(optimized_file)

    if baseline_results is None:
        print(f"❌ 无法加载标准版结果: {baseline_file}")
        sys.exit(1)

    if optimized_results is None:
        print(f"❌ 无法加载高召回率版结果: {optimized_file}")
        sys.exit(1)

    print_comparison(baseline_results, optimized_results)


if __name__ == "__main__":
    main()
