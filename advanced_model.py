#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级模型实现 - 包含决策树回归和集成方法
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import json

class DecisionTreeRegressor:
    """手动实现的决策树回归算法"""
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    class Node:
        def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None, 
                     left: Optional['DecisionTreeRegressor.Node'] = None, 
                     right: Optional['DecisionTreeRegressor.Node'] = None, 
                     value: Optional[float] = None, is_leaf: bool = False):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.is_leaf = is_leaf
    
    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """训练决策树"""
        n_samples, n_features = X.shape
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.all(y == y[0])):
            
            leaf_value = float(np.mean(y))
            self.tree = self.Node(value=leaf_value, is_leaf=True)
            return
        
        # 寻找最佳分割
        best_feature, best_threshold, best_score = self._find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = float(np.mean(y))
            self.tree = self.Node(value=leaf_value, is_leaf=True)
            return
        
        # 分割数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            leaf_value = float(np.mean(y))
            self.tree = self.Node(value=leaf_value, is_leaf=True)
            return
        
        # 创建节点
        self.tree = self.Node(feature_idx=best_feature, threshold=best_threshold)
        
        # 递归构建子树
        left_tree = DecisionTreeRegressor(self.max_depth, self.min_samples_split)
        right_tree = DecisionTreeRegressor(self.max_depth, self.min_samples_split)
        
        left_tree.fit(X[left_indices], y[left_indices], depth + 1)
        right_tree.fit(X[right_indices], y[right_indices], depth + 1)
        
        self.tree.left = left_tree.tree
        self.tree.right = right_tree.tree
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """寻找最佳分割点"""
        n_samples, n_features = X.shape
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        
        # 当前方差
        current_variance = np.var(y)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 为了效率，只考虑一些候选分割点
            if len(unique_values) > 10:
                percentiles = np.percentile(feature_values, [10, 25, 50, 75, 90])
                candidates = percentiles
            else:
                candidates = unique_values
            
            for threshold in candidates:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                # 计算加权方差
                n_left, n_right = len(left_indices), len(right_indices)
                var_left = np.var(y[left_indices]) if n_left > 1 else 0
                var_right = np.var(y[right_indices]) if n_right > 1 else 0
                
                weighted_variance = (n_left * var_left + n_right * var_right) / n_samples
                
                if weighted_variance < best_score:
                    best_score = weighted_variance
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_score
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.tree is None:
            raise ValueError("模型尚未训练")
        
        predictions = []
        for sample in X:
            predictions.append(self._predict_single(sample, self.tree))
        
        return np.array(predictions)
    
    def _predict_single(self, sample: np.ndarray, node: 'DecisionTreeRegressor.Node') -> float:
        """对单个样本进行预测"""
        if node.is_leaf:
            return float(node.value) if node.value is not None else 0.0
        
        if node.feature_idx is not None and node.threshold is not None:
            if sample[node.feature_idx] <= node.threshold:
                if node.left is not None:
                    return self._predict_single(sample, node.left)
            else:
                if node.right is not None:
                    return self._predict_single(sample, node.right)
        
        return 0.0  # 默认值

class GradientBoostingRegressor:
    """手动实现的梯度提升回归算法"""
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, 
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练梯度提升模型"""
        # 初始预测（均值）
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(len(y), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # 计算残差
            residuals = y - current_predictions
            
            # 训练决策树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # 更新预测
            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            
            # 打印进度
            if (i + 1) % 10 == 0:
                mse = np.mean((y - current_predictions) ** 2)
                print(f"Tree {i+1}/{self.n_estimators}, MSE: {mse:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.initial_prediction is None:
            raise ValueError("模型尚未训练")
        
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

class EnsembleModel:
    """集成模型，结合岭回归和梯度提升"""
    
    def __init__(self, ridge_weight: float = 0.5, gb_weight: float = 0.5):
        self.ridge_weight = ridge_weight
        self.gb_weight = gb_weight
        self.ridge_model = None
        self.gb_model = None
        self.is_trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练集成模型"""
        print("训练岭回归模型...")
        from model import RidgeRegression
        self.ridge_model = RidgeRegression(alpha=0.1)
        self.ridge_model.fit(X, y)
        
        print("训练梯度提升模型...")
        self.gb_model = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3)
        self.gb_model.fit(X, y)
        
        self.is_trained = True
        print("集成模型训练完成!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.ridge_model is None or self.gb_model is None:
            raise ValueError("模型组件未初始化")
        
        ridge_pred = self.ridge_model.predict(X)
        gb_pred = self.gb_model.predict(X)
        
        return self.ridge_weight * ridge_pred + self.gb_weight * gb_pred

def main():
    """测试高级模型"""
    print("开始测试高级模型...")
    
    # 加载数据
    df = pd.read_csv('train.csv')
    
    # 使用部分数据进行快速测试
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"使用样本大小: {sample_size}")
    
    # 特征工程
    from model import Model
    feature_engineer = Model().feature_engineer
    feature_engineer.fit(df_sample)
    X = feature_engineer.transform(df_sample)
    y = np.array(df_sample['age'].values)
    
    # 分割训练验证集
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    
    # 测试决策树
    print("\n测试决策树回归...")
    dt_model = DecisionTreeRegressor(max_depth=5)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_val)
    dt_rmse = np.sqrt(np.mean((y_val - dt_pred) ** 2))
    print(f"决策树RMSE: {dt_rmse:.4f}")
    
    # 测试梯度提升
    print("\n测试梯度提升回归...")
    gb_model = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1, max_depth=3)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_val)
    gb_rmse = np.sqrt(np.mean((y_val - gb_pred) ** 2))
    print(f"梯度提升RMSE: {gb_rmse:.4f}")
    
    # 测试集成模型
    print("\n测试集成模型...")
    ensemble_model = EnsembleModel(ridge_weight=0.6, gb_weight=0.4)
    ensemble_model.fit(X_train, y_train)
    ensemble_pred = ensemble_model.predict(X_val)
    ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_pred) ** 2))
    print(f"集成模型RMSE: {ensemble_rmse:.4f}")
    
    print("\n模型比较:")
    print(f"决策树:     {dt_rmse:.4f}")
    print(f"梯度提升:   {gb_rmse:.4f}")
    print(f"集成模型:   {ensemble_rmse:.4f}")

if __name__ == "__main__":
    main()