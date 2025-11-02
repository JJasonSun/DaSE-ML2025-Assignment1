#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试solution.py推理接口
"""

from solution import Solution
import time

def test_solution():
    """测试solution.py推理接口"""
    print("=== solution.py 推理测试 ===")
    
    # 初始化解决方案
    solution = Solution()
    
    # 测试样本
    test_sample = {
        'id': 666336, 
        'job': 'blue-collar', 
        'marital': 'married', 
        'education': 'secondary', 
        'default': 'no', 
        'balance': 3595,
        'housing': 'no', 
        'loan': 'yes', 
        'contact': 'unknown', 
        'day': 3, 
        'month': 'jul', 
        'duration': 198, 
        'campaign': 2,
        'pdays': -1, 
        'previous': 0, 
        'poutcome': 'unknown'
    }
    
    # 测试推理
    start_time = time.time()
    result = solution.forward(test_sample)
    end_time = time.time()
    
    print(f"预测年龄: {result['prediction']:.2f}")
    print(f"推理时间: {(end_time - start_time) * 1000:.2f}ms")
    print("✅ solution.py 推理接口测试通过!")
    
    return result

if __name__ == "__main__":
    test_solution()