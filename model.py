import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import json
import os

class FeatureEngineer:
    """特征工程类，处理数据预处理和特征生成"""
    
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_names = []
        self.categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                                   'loan', 'contact', 'month', 'poutcome']
        self.numerical_features = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        self.target_feature = 'age'
        
    def fit(self, df: pd.DataFrame):
        """基于训练数据拟合特征工程参数"""
        # 处理分类变量 - 目标编码
        for col in self.categorical_features:
            if col in df.columns:
                # 使用目标编码，用年龄的均值作为编码值
                target_mean = df.groupby(col)[self.target_feature].mean()
                self.categorical_encoders[col] = target_mean.to_dict()
        
        # 处理数值变量 - 标准化
        for col in self.numerical_features:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                self.numerical_scalers[col] = {'mean': mean_val, 'std': std_val}
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """应用特征工程转换"""
        features = []
        
        # 处理分类变量
        for col in self.categorical_features:
            if col in df.columns:
                encoded = df[col].map(self.categorical_encoders.get(col, {})).fillna(0)
                features.append(encoded.values)
        
        # 处理数值变量
        for col in self.numerical_features:
            if col in df.columns:
                scaler = self.numerical_scalers.get(col, {'mean': 0, 'std': 1})
                normalized = (df[col] - scaler['mean']) / (scaler['std'] + 1e-8)
                features.append(normalized.values)
        
        # 生成多项式特征（二阶交互项）
        base_features = np.array(features).T
        poly_features = []
        
        # 添加原始特征
        for i in range(base_features.shape[1]):
            poly_features.append(base_features[:, i])
        
        # 添加二阶交互项
        n_features = base_features.shape[1]
        for i in range(n_features):
            for j in range(i, n_features):
                interaction = base_features[:, i] * base_features[:, j]
                poly_features.append(interaction)
        
        # 添加平方项
        for i in range(n_features):
            squared = base_features[:, i] ** 2
            poly_features.append(squared)
        
        return np.array(poly_features).T
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        names = []
        
        # 原始特征名
        for col in self.categorical_features:
            names.append(f"{col}_encoded")
        for col in self.numerical_features:
            names.append(f"{col}_scaled")
        
        # 多项式特征名
        base_names = names.copy()
        n_features = len(base_names)
        
        # 交互项
        for i in range(n_features):
            for j in range(i, n_features):
                names.append(f"{base_names[i]}_x_{base_names[j]}")
        
        # 平方项
        for i in range(n_features):
            names.append(f"{base_names[i]}_squared")
        
        return names

class SHAPAnalyzer:
    """手动实现的SHAP分析器"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def kernel_shap(self, X_background: np.ndarray, X_explain: np.ndarray, 
                   nsamples: int = 100) -> np.ndarray:
        """KernelSHAP实现"""
        n_samples, n_features = X_explain.shape
        shap_values = np.zeros((n_samples, n_features))
        
        # 背景数据期望值
        f_background = self.model.predict(X_background)
        expected_value = np.mean(f_background)
        
        for i in range(n_samples):
            x = X_explain[i:i+1]
            
            # 生成联盟子集
            coalitions = []
            weights = []
            
            for _ in range(nsamples):
                # 随机生成联盟掩码
                mask = np.random.random(n_features) < 0.5
                coalitions.append(mask)
                
                # 计算SHAP权重
                z = np.sum(mask)
                if z == 0 or z == n_features:
                    weight = 1e6
                else:
                    weight = (n_features - 1) / (z * (n_features - z))
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 计算每个联盟的贡献
            f_coalitions = []
            for mask in coalitions:
                # 创建联盟特征
                x_coalition = x.copy()
                for j in range(n_features):
                    if not mask[j]:
                        # 使用背景数据的随机样本替换
                        bg_idx = np.random.randint(0, len(X_background))
                        x_coalition[0, j] = X_background[bg_idx, j]
                
                f_coalitions.append(self.model.predict(x_coalition)[0])
            
            f_coalitions = np.array(f_coalitions)
            
            # 计算SHAP值
            for j in range(n_features):
                marginal_contributions = []
                marginal_weights = []
                
                for k, mask in enumerate(coalitions):
                    if mask[j]:
                        # 计算边际贡献
                        mask_without_j = mask.copy()
                        mask_without_j[j] = False
                        
                        # 创建不包含特征j的联盟
                        x_without_j = x.copy()
                        for l in range(n_features):
                            if not mask_without_j[l]:
                                bg_idx = np.random.randint(0, len(X_background))
                                x_without_j[0, l] = X_background[bg_idx, l]
                        
                        f_with_j = f_coalitions[k]
                        f_without_j = self.model.predict(x_without_j)[0]
                        
                        marginal_contributions.append(f_with_j - f_without_j)
                        marginal_weights.append(weights[k])
                
                if marginal_contributions:
                    shap_values[i, j] = np.average(marginal_contributions, 
                                                  weights=marginal_weights)
        
        return shap_values
    
    def get_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """计算特征重要性"""
        importance = np.mean(np.abs(shap_values), axis=0)
        return dict(zip(self.feature_names, importance))

class RidgeRegression:
    """手动实现的岭回归算法"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练岭回归模型"""
        n_samples, n_features = X.shape
        
        # 添加偏置项
        X_with_bias = np.column_stack([X, np.ones(n_samples)])
        
        # 计算岭回归的闭式解
        # w = (X^T X + αI)^(-1) X^T y
        identity_matrix = np.eye(n_features + 1)
        identity_matrix[-1, -1] = 0  # 不对偏置项进行正则化
        
        XTX = np.dot(X_with_bias.T, X_with_bias)
        XTy = np.dot(X_with_bias.T, y)
        
        # 添加正则化项
        XTX_regularized = XTX + self.alpha * identity_matrix
        
        # 求解权重
        try:
            weights_with_bias = np.linalg.solve(XTX_regularized, XTy)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            weights_with_bias = np.linalg.pinv(XTX_regularized).dot(XTy)
        
        self.weights = weights_with_bias[:-1]
        self.bias = weights_with_bias[-1]
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.weights is None:
            raise ValueError("模型权重未初始化")
        return np.dot(X, self.weights) + self.bias
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {'alpha': self.alpha, 'weights': self.weights, 'bias': self.bias}

class Model:
    def __init__(self):
        """初始化模型参数"""
        self.feature_engineer = FeatureEngineer()
        self.ridge_model = RidgeRegression(alpha=1.0)
        self.shap_analyzer = None
        self.feature_importance = {}
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """训练模型"""
        print("开始特征工程...")
        self.feature_engineer.fit(df)
        
        print("生成特征矩阵...")
        X = self.feature_engineer.transform(df)
        y = np.array(df['age'].values)
        
        print(f"特征矩阵形状: {X.shape}")
        
        # 数据分割用于SHAP分析
        n_background = min(100, len(X) // 10)
        X_background = X[:n_background]
        X_train = X[n_background:]
        y_train = y[n_background:]
        
        print("训练岭回归模型...")
        self.ridge_model.fit(X_train, y_train)
        
        # 计算训练误差
        train_pred = self.ridge_model.predict(X_train)
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        print(f"训练RMSE: {train_rmse:.4f}")
        
        print("进行SHAP分析...")
        feature_names = self.feature_engineer.get_feature_names()
        self.shap_analyzer = SHAPAnalyzer(self.ridge_model, feature_names)
        
        # 对部分样本进行SHAP分析
        n_explain = min(50, len(X_train))
        shap_values = self.shap_analyzer.kernel_shap(X_background, X_train[:n_explain], 
                                                    nsamples=50)
        self.feature_importance = self.shap_analyzer.get_feature_importance(shap_values)
        
        print("特征重要性排序:")
        sorted_importance = sorted(self.feature_importance.items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_importance[:10]:
            print(f"  {feature}: {importance:.6f}")
        
        self.is_trained = True
        print("模型训练完成!")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.ridge_model.predict(X)
    
    def predict_single(self, sample_dict: Dict[str, Any]) -> float:
        """对单个样本进行预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 转换为DataFrame
        df = pd.DataFrame([sample_dict])
        
        # 特征工程
        X = self.feature_engineer.transform(df)
        
        # 预测
        prediction = self.ridge_model.predict(X)[0]
        
        return float(prediction)
    
    def save_model(self, filepath: str):
        """保存模型"""
        ridge_params = self.ridge_model.get_params()
        model_data = {
            'feature_engineer': {
                'categorical_encoders': self.feature_engineer.categorical_encoders,
                'numerical_scalers': self.feature_engineer.numerical_scalers,
                'feature_names': self.feature_engineer.feature_names
            },
            'ridge_params': {
                'alpha': ridge_params['alpha'],
                'weights': ridge_params['weights'].tolist() if ridge_params['weights'] is not None else None,
                'bias': float(ridge_params['bias']) if ridge_params['bias'] is not None else None
            },
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # 恢复特征工程器
        self.feature_engineer.categorical_encoders = model_data['feature_engineer']['categorical_encoders']
        self.feature_engineer.numerical_scalers = model_data['feature_engineer']['numerical_scalers']
        self.feature_engineer.feature_names = model_data['feature_engineer']['feature_names']
        
        # 恢复岭回归模型
        self.ridge_model.alpha = model_data['ridge_params']['alpha']
        weights_data = model_data['ridge_params']['weights']
        self.ridge_model.weights = np.array(weights_data) if weights_data is not None else None
        bias_data = model_data['ridge_params']['bias']
        self.ridge_model.bias = bias_data
        
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']