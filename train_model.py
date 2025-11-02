#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本
"""

import pandas as pd
import numpy as np
from model import Model
import warnings
warnings.filterwarnings('ignore')

def main():
    print("开始加载数据...")
    
    # 读取训练数据
    df = pd.read_csv('train.csv')
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查数据
    print("\n数据预览:")
    print(df.head())
    
    print("\n数据统计信息:")
    print(df.describe())
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 初始化模型
    print("\n初始化模型...")
    model = Model()
    
    # 训练模型
    print("开始训练模型...")
    model.train(df)
    
    # 保存模型
    model_path = 'trained_model.json'
    model.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 在训练集上评估
    print("\n在训练集上评估...")
    X = model.feature_engineer.transform(df)
    y_true = df['age'].values
    y_pred = model.predict(X)
    
    # 计算评估指标
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 显示一些预测结果
    print("\n预测结果示例:")
    for i in range(10):
        print(f"真实年龄: {y_true[i]:.1f}, 预测年龄: {y_pred[i]:.1f}, 误差: {abs(y_true[i] - y_pred[i]):.1f}")

if __name__ == "__main__":
    main()