# -*- coding:utf-8 -*-
"""
初始化指定号码配置
"""

import json
import os

def init_specified_numbers():
    """初始化指定号码配置"""
    
    # 您指定的号码
    specified_config = {
        "specified_red_numbers": [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        "specified_blue_numbers": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "red_zones": {
            'zone1': [4, 5, 6, 7, 9, 10, 11, 12],
            'zone2': [13, 14, 17, 18, 19, 20, 21, 25],
            'zone3': [26, 27, 28, 29, 30, 31, 32, 33, 34]
        },
        "blue_zones": {
            'zone1': [2, 3, 4, 5],
            'zone2': [6, 7, 8],
            'zone3': [9, 10, 11]
        }
    }
    
    # 保存配置
    config_file = "specified_numbers_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(specified_config, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("指定号码配置已初始化")
    print("=" * 60)
    print(f"红球指定号码 ({len(specified_config['specified_red_numbers'])}个):")
    print(f"  {sorted(specified_config['specified_red_numbers'])}")
    print()
    print(f"蓝球指定号码 ({len(specified_config['specified_blue_numbers'])}个):")
    print(f"  {sorted(specified_config['specified_blue_numbers'])}")
    print()
    print("红球分区:")
    for zone, numbers in specified_config['red_zones'].items():
        print(f"  {zone}: {numbers}")
    print()
    print("蓝球分区:")
    for zone, numbers in specified_config['blue_zones'].items():
        print(f"  {zone}: {numbers}")
    print("=" * 60)
    print(f"配置已保存到: {config_file}")
    
    # 创建用于导入的Python文件
    py_config_file = "specified_numbers.py"
    with open(py_config_file, 'w', encoding='utf-8') as f:
        f.write("# -*- coding:utf-8 -*-\n")
        f.write('"""\n指定号码配置\n"""\n\n')
        f.write(f"SPECIFIED_RED_NUMBERS = {specified_config['specified_red_numbers']}\n")
        f.write(f"SPECIFIED_BLUE_NUMBERS = {specified_config['specified_blue_numbers']}\n")
    
    print(f"Python配置文件已保存到: {py_config_file}")
    
    return specified_config

if __name__ == "__main__":
    init_specified_numbers()