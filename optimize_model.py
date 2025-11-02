#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型参数优化脚本
"""

import pandas as pd
import numpy as np
from model import Model, RidgeRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def cross_validate_ridge(X, y, alpha, n_folds=5):
    """交叉验证岭回归模型"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练岭回归
        model = RidgeRegression(alpha=alpha)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores), np.std(rmse_scores)

def optimize_alpha(X, y, alphas):
    """寻找最佳alpha参数"""
    best_alpha = None
    best_score = float('inf')
    results = []
    
    print("开始参数调优...")
    for alpha in alphas:
        mean_rmse, std_rmse = cross_validate_ridge(X, y, alpha)
        results.append((alpha, mean_rmse, std_rmse))
        print(f"Alpha: {alpha:.6f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_alpha = alpha
    
    print(f"\n最佳Alpha: {best_alpha:.6f}, 最佳RMSE: {best_score:.4f}")
    return best_alpha, results

def main():
    print("开始加载数据...")
    df = pd.read_csv('train.csv')
    print(f"数据形状: {df.shape}")
    
    # 使用部分数据进行快速调优
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"使用样本大小: {sample_size}")
    
    # 特征工程
    print("进行特征工程...")
    feature_engineer = Model().feature_engineer
    feature_engineer.fit(df_sample)
    X = feature_engineer.transform(df_sample)
    y = np.array(df_sample['age'].values)
    
    # 定义alpha搜索范围
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # 参数调优
    best_alpha, results = optimize_alpha(X, y, alphas)
    
    # 使用最佳参数训练完整模型
    print("\n使用最佳参数训练完整模型...")
    full_model = Model()
    # 创建新的岭回归模型实例，使用最佳alpha
    if best_alpha is not None:
        full_model.ridge_model = RidgeRegression(alpha=float(best_alpha))
    else:
        full_model.ridge_model = RidgeRegression(alpha=1.0)
    full_model.train(df)
    
    # 保存优化后的模型
    full_model.save_model('optimized_model.json')
    print("优化后的模型已保存到: optimized_model.json")
    
    # 在完整训练集上评估
    X_full = full_model.feature_engineer.transform(df)
    y_true = np.array(df['age'].values)
    y_pred = full_model.predict(X_full)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\n最终模型性能:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return best_alpha, rmse

if __name__ == "__main__":
    best_alpha, final_rmse = main()
    print(f"\n优化完成！最佳Alpha: {best_alpha}, 最终RMSE: {final_rmse:.4f}")