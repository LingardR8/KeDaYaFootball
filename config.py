# -*- coding:utf-8 -*-
"""
Author: BigCat
Enhanced with AC Value, Zone Analysis, and Genetic Algorithm
Fixed path configuration issues
修改为指定号码集合：红球25个，蓝球10个
"""
import os

# 获取当前工作目录并构建正确的模型路径
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "model")

# 确保模型目录存在
if not os.path.exists(model_path):
    os.makedirs(model_path)

ball_name = [
    ("红球", "red"),
    ("蓝球", "blue")
]

data_file_name = "data.csv"

name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/"
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/"
    }
}

# ========== 您指定的号码集合 ==========
SPECIFIED_RED_NUMBERS = [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
SPECIFIED_BLUE_NUMBERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 排序号码
SPECIFIED_RED_NUMBERS.sort()
SPECIFIED_BLUE_NUMBERS.sort()

# 基于指定号码创建区间划分
def create_zones_for_specified_numbers(numbers, num_zones):
    """为指定号码创建区间划分"""
    numbers = sorted(numbers)
    zone_size = len(numbers) // num_zones
    remainder = len(numbers) % num_zones
    zones = {}
    
    start = 0
    for i in range(num_zones):
        end = start + zone_size
        if i < remainder:
            end += 1
        zones[f'zone{i+1}'] = numbers[start:end]
        start = end
    
    return zones

# 创建红球和蓝球的区间划分
# 双色球红球分为3区，蓝球分为3区
red_zones_ssq = create_zones_for_specified_numbers(SPECIFIED_RED_NUMBERS, 3)
blue_zones_ssq = create_zones_for_specified_numbers(SPECIFIED_BLUE_NUMBERS, 3)

# 大乐透红球分为5区，蓝球分为3区
red_zones_dlt = create_zones_for_specified_numbers(SPECIFIED_RED_NUMBERS, 5)
blue_zones_dlt = create_zones_for_specified_numbers(SPECIFIED_BLUE_NUMBERS, 3)

zone_configs = {
    "ssq": {
        "red_zones": red_zones_ssq,
        "blue_zones": blue_zones_ssq
    },
    "dlt": {
        "red_zones": red_zones_dlt,
        "blue_zones": blue_zones_dlt
    }
}

# AC值算法配置 - 基于指定号码范围重新计算
ac_value_configs = {
    "ssq": {
        "red": {"min_ac": 4, "max_ac": 8, "optimal_ac": 6},
        "blue": {"min_ac": 1, "max_ac": 3, "optimal_ac": 2}
    },
    "dlt": {
        "red": {"min_ac": 5, "max_ac": 9, "optimal_ac": 7},
        "blue": {"min_ac": 0, "max_ac": 2, "optimal_ac": 1}
    }
}

# 遗传算法配置
genetic_algorithm_configs = {
    "population_size": 50,
    "generations": 20,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "elite_count": 5
}

# 确保每个模型子目录存在
def ensure_model_dirs():
    """确保所有模型目录存在"""
    for name in ["ssq", "dlt"]:
        for ball_type in ["red", "blue"]:
            dir_path = os.path.join(model_path, name, f"{ball_type}_ball_model")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"创建目录: {dir_path}")

# 调用函数确保目录存在
ensure_model_dirs()

model_args = {
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "sequence_len": 6,
            "red_n_class": 25,    # 红球25个指定号码
            "red_epochs": 14,
            "red_embedding_size": 320,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_n_class": 10,   # 蓝球10个指定号码
            "blue_epochs": 14,
            "blue_embedding_size": 640,
            "blue_hidden_size": 32,
            "blue_layer_size": 1,
            # 复式参数
            "red_output_count": 13,
            "blue_output_count": 5,
            # 算法增强参数
            "ac_weight": 0.1,  # AC值正则化权重
            "zone_weight": 0.1,  # 断区正则化权重
            "enable_algorithm": True,  # 启用算法增强
            # 指定号码参数
            "specified_red_numbers": SPECIFIED_RED_NUMBERS,
            "specified_blue_numbers": SPECIFIED_BLUE_NUMBERS
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": os.path.join(model_path, "ssq", "red_ball_model") + os.sep,
            "blue": os.path.join(model_path, "ssq", "blue_ball_model") + os.sep
        },
        "algorithm_config": {
            "zone_config": zone_configs["ssq"],
            "ac_config": ac_value_configs["ssq"],
            "ga_config": genetic_algorithm_configs
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 10,
            "batch_size": 1,
            "custom_red_num": 10,
            "red_sequence_len": 5,
            "red_n_class": 25,    # 红球25个指定号码
            "red_epochs": 14,
            "red_embedding_size": 256,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 2,
            "blue_n_class": 10,   # 蓝球10个指定号码
            "blue_epochs": 14,
            "blue_embedding_size": 128,
            "blue_hidden_size": 20,
            "blue_layer_size": 1,
            # 复式参数
            "red_output_count": 13,
            "blue_output_count": 5,
            # 算法增强参数
            "ac_weight": 0.3,  # AC值正则化权重
            "zone_weight": 0.4,  # 断区正则化权重
            "enable_algorithm": True,  # 启用算法增强
            # 指定号码参数
            "specified_red_numbers": SPECIFIED_RED_NUMBERS,
            "specified_blue_numbers": SPECIFIED_BLUE_NUMBERS
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": os.path.join(model_path, "dlt", "red_ball_model") + os.sep,
            "blue": os.path.join(model_path, "dlt", "blue_ball_model") + os.sep
        },
        "algorithm_config": {
            "zone_config": zone_configs["dlt"],
            "ac_config": ac_value_configs["dlt"],
            "ga_config": genetic_algorithm_configs
        }
    }
}

# 模型名
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"


# 添加路径验证函数
def validate_paths():
    """验证所有配置路径是否存在"""
    print("=" * 60)
    print("指定号码配置验证")
    print("=" * 60)
    print(f"红球指定号码 ({len(SPECIFIED_RED_NUMBERS)}个):")
    print(f"  {SPECIFIED_RED_NUMBERS}")
    print()
    print(f"蓝球指定号码 ({len(SPECIFIED_BLUE_NUMBERS)}个):")
    print(f"  {SPECIFIED_BLUE_NUMBERS}")
    print()
    print("红球分区 (大乐透):")
    for zone, numbers in red_zones_dlt.items():
        print(f"  {zone}: {numbers}")
    print()
    print("蓝球分区 (大乐透):")
    for zone, numbers in blue_zones_dlt.items():
        print(f"  {zone}: {numbers}")
    print("=" * 60)
    
    print(f"模型根目录: {model_path}")
    print(f"模型根目录存在: {os.path.exists(model_path)}")

    for name in ["ssq", "dlt"]:
        print(f"\n{name_path[name]['name']}路径验证:")
        for ball_type in ["red", "blue"]:
            path = model_args[name]["path"][ball_type]
            exists = os.path.exists(path)
            print(f"  {ball_type}路径: {path}")
            print(f"  路径存在: {exists}")

            if exists:
                files = os.listdir(path)
                print(f"  目录内容: {files[:5]}...")  # 只显示前5个文件
            else:
                print(f"  警告: 路径不存在，将自动创建")


# 在导入时验证路径
if __name__ == "__main__":
    validate_paths()
else:
    # 在模块导入时确保目录存在
    ensure_model_dirs()