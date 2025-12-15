# -*- coding:utf-8 -*-
"""
过滤历史数据，只保留指定号码的开奖记录
"""

import pandas as pd
import numpy as np
import os
from config import SPECIFIED_RED_NUMBERS, SPECIFIED_BLUE_NUMBERS

def filter_historical_data(name="dlt"):
    """过滤历史数据，只保留指定号码的开奖记录"""
    
    # 读取原始数据
    data_path = f"data/{name}/data.csv"
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"原始数据期数: {len(df)}")
    
    # 检查号码列
    if name == "ssq":
        red_columns = [f"红球_{i+1}" for i in range(6)]
        blue_columns = ["蓝球_1"]
    else:
        red_columns = [f"红球_{i+1}" for i in range(5)]
        blue_columns = [f"蓝球_{i+1}" for i in range(2)]
    
    print(f"红球列: {red_columns}")
    print(f"蓝球列: {blue_columns}")
    
    # 过滤数据
    valid_rows = []
    for idx, row in df.iterrows():
        red_valid = all(row[col] in SPECIFIED_RED_NUMBERS for col in red_columns)
        blue_valid = all(row[col] in SPECIFIED_BLUE_NUMBERS for col in blue_columns)
        
        if red_valid and blue_valid:
            valid_rows.append(idx)
    
    filtered_df = df.loc[valid_rows].copy()
    
    # 保存过滤后的数据
    filtered_path = f"data/{name}/data_filtered.csv"
    filtered_df.to_csv(filtered_path, index=False)
    
    print("=" * 60)
    print("数据过滤结果:")
    print("=" * 60)
    print(f"指定红球号码: {sorted(SPECIFIED_RED_NUMBERS)}")
    print(f"指定蓝球号码: {sorted(SPECIFIED_BLUE_NUMBERS)}")
    print(f"原始数据期数: {len(df)}")
    print(f"过滤后数据期数: {len(filtered_df)}")
    print(f"过滤比例: {len(filtered_df)/len(df)*100:.2f}%")
    print(f"过滤后数据已保存到: {filtered_path}")
    
    # 显示前几期数据
    if len(filtered_df) > 0:
        print("\n过滤后数据示例:")
        print(filtered_df.head())
    
    return filtered_df

if __name__ == "__main__":
    # 过滤大乐透数据
    filter_historical_data("dlt")