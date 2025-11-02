import pandas as pd
import numpy as np
from typing import Dict, Any
from model import Model

class Solution:        
    def forward(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """模型推理接口，接收单条样本数据并返回预测结果
        
        Args:
            sample: 单条样本数据字典，包含ID列及特征列（不含age列）
                示例: {'id': 666336, 'job': 'blue-collar', 'marital': 'married', 
                       'education': 'secondary', 'default': 'no', 'balance': 3595,
                       'housing': 'no', 'loan': 'yes', 'contact': 'unknown', 
                       'day': 3, 'month': 'jul', 'duration': 198, 'campaign': 2,
                       'pdays': -1, 'previous': 0, 'poutcome': 'unknown'}
        
        Returns:
            包含预测结果的字典，格式为: {'prediction': 预测年龄值}
        """
        # 初始化模型（延迟加载）
        if not hasattr(self, '_model_initialized'):
            self.model = Model()
            try:
                # 优先加载优化后的模型
                self.model.load_model('optimized_model.json')
            except FileNotFoundError:
                try:
                    # 如果没有优化模型，加载基础模型
                    self.model.load_model('trained_model.json')
                except FileNotFoundError:
                    print("警告: 未找到预训练模型，请先运行train_model.py训练模型")
                    self.model = None
            self._model_initialized = True
        
        if self.model is None:
            # 如果模型未加载，返回默认值
            return {'prediction': 30.0}
        
        try:
            # 确保样本不包含age列（这是我们要预测的目标）
            if 'age' in sample:
                sample = sample.copy()
                del sample['age']
            
            # 使用模型的单样本预测方法
            prediction = self.model.predict_single(sample)
            
            return {'prediction': float(prediction)}
            
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            # 返回一个合理的默认值
            return {'prediction': 30.0}