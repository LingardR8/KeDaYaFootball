# -*- coding:utf-8 -*-
"""
验证所有功能都使用指定号码
"""

from config import SPECIFIED_RED_NUMBERS, SPECIFIED_BLUE_NUMBERS
import json
import os

def verify_configuration():
    """验证配置是否正确"""
    
    print("验证指定号码配置:")
    print("=" * 50)
    
    # 验证红球
    expected_red = [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    red_correct = set(SPECIFIED_RED_NUMBERS) == set(expected_red)
    
    print(f"红球号码: {'✓' if red_correct else '✗'}")
    print(f"  配置: {sorted(SPECIFIED_RED_NUMBERS)}")
    print(f"  预期: {sorted(expected_red)}")
    
    # 验证蓝球
    expected_blue = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    blue_correct = set(SPECIFIED_BLUE_NUMBERS) == set(expected_blue)
    
    print(f"蓝球号码: {'✓' if blue_correct else '✗'}")
    print(f"  配置: {sorted(SPECIFIED_BLUE_NUMBERS)}")
    print(f"  预期: {sorted(expected_blue)}")
    
    # 验证数量
    red_count_correct = len(SPECIFIED_RED_NUMBERS) == 25
    blue_count_correct = len(SPECIFIED_BLUE_NUMBERS) == 10
    
    print(f"红球数量: {len(SPECIFIED_RED_NUMBERS)} (预期: 25) {'✓' if red_count_correct else '✗'}")
    print(f"蓝球数量: {len(SPECIFIED_BLUE_NUMBERS)} (预期: 10) {'✓' if blue_count_correct else '✗'}")
    
    # 验证没有重复
    red_unique = len(SPECIFIED_RED_NUMBERS) == len(set(SPECIFIED_RED_NUMBERS))
    blue_unique = len(SPECIFIED_BLUE_NUMBERS) == len(set(SPECIFIED_BLUE_NUMBERS))
    
    print(f"红球无重复: {'✓' if red_unique else '✗'}")
    print(f"蓝球无重复: {'✓' if blue_unique else '✗'}")
    
    # 验证配置文件
    config_file = "config.py"
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        has_red_config = "SPECIFIED_RED_NUMBERS" in content
        has_blue_config = "SPECIFIED_BLUE_NUMBERS" in content
        
        print(f"配置文件包含红球配置: {'✓' if has_red_config else '✗'}")
        print(f"配置文件包含蓝球配置: {'✓' if has_blue_config else '✗'}")
    else:
        print("配置文件不存在: ✗")
    
    print("=" * 50)
    
    all_correct = (red_correct and blue_correct and red_unique and blue_unique and 
                   red_count_correct and blue_count_correct)
    
    if all_correct:
        print("✓ 配置验证通过！")
        return True
    else:
        print("✗ 配置验证失败！")
        return False

def verify_data_filtering():
    """验证数据过滤功能"""
    print("\n验证数据过滤功能:")
    print("=" * 50)
    
    try:
        from filter_historical_data import filter_historical_data
        filtered_df = filter_historical_data("dlt")
        
        if filtered_df is not None and len(filtered_df) > 0:
            print("✓ 数据过滤功能正常")
            
            # 检查过滤后的数据是否都包含指定号码
            all_valid = True
            for idx, row in filtered_df.iterrows():
                red_balls = [row[f"红球_{i+1}"] for i in range(5)]
                blue_balls = [row[f"蓝球_{i+1}"] for i in range(2)]
                
                red_valid = all(ball in SPECIFIED_RED_NUMBERS for ball in red_balls)
                blue_valid = all(ball in SPECIFIED_BLUE_NUMBERS for ball in blue_balls)
                
                if not (red_valid and blue_valid):
                    all_valid = False
                    print(f"  第{idx}期数据包含非指定号码")
                    break
            
            print(f"  过滤后数据有效性: {'✓' if all_valid else '✗'}")
            return all_valid
        else:
            print("✗ 数据过滤失败或没有有效数据")
            return False
            
    except Exception as e:
        print(f"✗ 数据过滤验证失败: {e}")
        return False

if __name__ == "__main__":
    config_ok = verify_configuration()
    print()
    data_ok = verify_data_filtering()
    
    print("\n" + "=" * 50)
    if config_ok and data_ok:
        print("✓ 所有验证通过！系统已配置为使用指定号码。")
    else:
        print("✗ 验证失败，请检查配置。")