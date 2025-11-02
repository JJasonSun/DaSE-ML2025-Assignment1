#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估和可视化脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model_performance():
    """评估模型性能"""
    print("=== 模型性能评估 ===")
    
    # 加载数据
    df = pd.read_csv('train.csv')
    print(f"数据集大小: {df.shape}")
    
    # 加载训练好的模型
    model = Model()
    try:
        model.load_model('optimized_model.json')
        print("使用优化模型")
    except FileNotFoundError:
        model.load_model('trained_model.json')
        print("使用基础模型")
    
    # 预测
    X = model.feature_engineer.transform(df)
    y_true = df['age'].values
    y_pred = model.predict(X)
    
    # 计算评估指标
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\n评估指标:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 年龄段分析
    age_groups = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                       labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df['age_group'] = age_groups
    
    group_rmse = []
    group_names = []
    
    for name, group in df.groupby('age_group'):
        group_indices = group.index.tolist()
        group_true = y_true[group_indices]
        group_pred = y_pred[group_indices]
        group_rmse_val = np.sqrt(np.mean((group_true - group_pred) ** 2))
        group_rmse.append(group_rmse_val)
        group_names.append(name)
        print(f"年龄段 {name}: RMSE = {group_rmse_val:.4f}")
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse,
        'group_rmse': group_rmse, 'group_names': group_names,
        'y_true': y_true, 'y_pred': y_pred, 'df': df
    }

def visualize_results(results):
    """可视化结果"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    df = results['df']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测vs真实值散点图
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实年龄')
    axes[0, 0].set_ylabel('预测年龄')
    axes[0, 0].set_title('预测值 vs 真实值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差分布
    residuals = y_true - y_pred
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('残差 (真实值 - 预测值)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('残差分布')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # 3. 年龄段RMSE
    group_names = results['group_names']
    group_rmse = results['group_rmse']
    bars = axes[1, 0].bar(group_names, group_rmse, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('年龄段')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('各年龄段预测误差')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar, rmse in zip(bars, group_rmse):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{rmse:.2f}', ha='center', va='bottom')
    
    # 4. 误差vs年龄
    age_bins = np.arange(15, 100, 5)
    age_centers = (age_bins[:-1] + age_bins[1:]) / 2
    age_rmse = []
    
    for i in range(len(age_bins) - 1):
        mask = (y_true >= age_bins[i]) & (y_true < age_bins[i + 1])
        if np.sum(mask) > 0:
            age_rmse.append(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        else:
            age_rmse.append(0)
    
    axes[1, 1].plot(age_centers, age_rmse, 'o-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('年龄')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('预测误差随年龄变化')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为 'model_evaluation.png'")

def analyze_feature_importance():
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    
    # 加载模型
    model = Model()
    try:
        model.load_model('optimized_model.json')
    except FileNotFoundError:
        model.load_model('trained_model.json')
    
    if not model.feature_importance:
        print("模型中没有SHAP特征重要性数据")
        return
    
    # 获取特征重要性
    importance = model.feature_importance
    sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n前20个最重要特征:")
    for i, (feature, imp) in enumerate(sorted_importance[:20]):
        print(f"{i+1:2d}. {feature:30s}: {abs(imp):8.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = sorted_importance[:15]
    features = [f[0] for f in top_features]
    importances = [abs(f[1]) for f in top_features]
    
    plt.barh(range(len(features)), importances, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.yticks(range(len(features)), features)
    plt.xlabel('绝对SHAP值')
    plt.title('特征重要性 (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("特征重要性图表已保存为 'feature_importance.png'")

def test_prediction_speed():
    """测试预测速度"""
    print("\n=== 预测速度测试 ===")
    
    import time
    
    # 加载模型
    model = Model()
    try:
        model.load_model('optimized_model.json')
    except FileNotFoundError:
        model.load_model('trained_model.json')
    
    # 准备测试数据
    df = pd.read_csv('train.csv')
    test_samples = df.sample(n=1000, random_state=42)
    
    # 特征工程
    X = model.feature_engineer.transform(test_samples)
    
    # 测试批量预测速度
    start_time = time.time()
    predictions = model.predict(X)
    batch_time = time.time() - start_time
    
    print(f"批量预测1000个样本耗时: {batch_time:.4f}秒")
    print(f"平均每个样本预测时间: {batch_time/1000*1000:.4f}毫秒")
    
    # 测试单样本预测速度
    sample_dict = test_samples.iloc[0].to_dict()
    del sample_dict['age']  # 移除目标变量
    
    start_time = time.time()
    for _ in range(100):
        _ = model.predict_single(sample_dict)
    single_time = (time.time() - start_time) / 100
    
    print(f"单样本预测平均耗时: {single_time*1000:.4f}毫秒")

def main():
    """主函数"""
    print("开始模型评估...")
    
    # 评估模型性能
    results = evaluate_model_performance()
    
    # 可视化结果
    try:
        visualize_results(results)
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
    
    # 分析特征重要性
    analyze_feature_importance()
    
    # 测试预测速度
    test_prediction_speed()
    
    print("\n=== 评估完成 ===")
    print("生成的文件:")
    print("- model_evaluation.png: 模型性能可视化")
    print("- feature_importance.png: 特征重要性可视化")

if __name__ == "__main__":
    main()