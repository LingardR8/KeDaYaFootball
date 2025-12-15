# -*- coding:utf-8 -*-
"""
Author: BigCat
Enhanced with AC Value, Zone Analysis, and Genetic Algorithm
Fixed Index Error: indices[0,0] = -30 is not in [0, 1089]
Fixed Dimension Mismatch Error in Genetic Algorithm
修改为使用指定号码集合进行训练和预测
"""
import time
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from config import *
from modeling import EnhancedLstmWithCRFModel, EnhancedSignalLstmModel
from loguru import logger
from datetime import datetime
import random
from typing import List, Dict, Tuple

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
parser.add_argument('--train_test_split', default=0.65, type=float, help="训练集占比, 设置大于0.5")
args = parser.parse_args()

pred_key = {}


class AlgorithmEnhancer:
    """算法增强器 - 集成AC值、断区和遗传算法，使用指定号码集合"""
    
    def __init__(self, name: str):
        self.name = name
        self.m_args = model_args[name]
        self.ac_config = self.m_args["algorithm_config"]["ac_config"]
        self.zone_config = self.m_args["algorithm_config"]["zone_config"]
        self.ga_config = self.m_args["algorithm_config"]["ga_config"]
        
        # 使用指定号码
        self.specified_red_numbers = sorted(SPECIFIED_RED_NUMBERS)
        self.specified_blue_numbers = sorted(SPECIFIED_BLUE_NUMBERS)
        
        # 创建映射字典（号码 -> 索引）
        self.red_mapping = {num: idx for idx, num in enumerate(self.specified_red_numbers)}
        self.blue_mapping = {num: idx for idx, num in enumerate(self.specified_blue_numbers)}
        
        # 反向映射（索引 -> 号码）
        self.red_reverse_mapping = {idx: num for idx, num in enumerate(self.specified_red_numbers)}
        self.blue_reverse_mapping = {idx: num for idx, num in enumerate(self.specified_blue_numbers)}
        
        logger.info(f"初始化指定号码 - 红球: {self.specified_red_numbers}")
        logger.info(f"初始化指定号码 - 蓝球: {self.specified_blue_numbers}")

    def map_to_specified_index(self, numbers: List[int], ball_type: str) -> List[int]:
        """将实际号码映射到指定号码的索引"""
        if ball_type == "red":
            mapping = self.red_mapping
        else:
            mapping = self.blue_mapping
            
        try:
            indices = [mapping[num] for num in numbers]
            return indices
        except KeyError as e:
            logger.error(f"号码 {e} 不在指定号码集合中")
            # 如果不在指定集合中，使用最接近的号码
            result = []
            for num in numbers:
                if num in mapping:
                    result.append(mapping[num])
                else:
                    # 找到最接近的指定号码
                    closest_num = min(self.specified_red_numbers if ball_type == "red" else self.specified_blue_numbers,
                                     key=lambda x: abs(x - num))
                    result.append(mapping[closest_num])
                    logger.warning(f"号码 {num} 不在指定集合中，使用最接近的 {closest_num}")
            return result

    def map_from_specified_index(self, indices: List[int], ball_type: str) -> List[int]:
        """从指定号码索引映射回实际号码"""
        if ball_type == "red":
            reverse_mapping = self.red_reverse_mapping
        else:
            reverse_mapping = self.blue_reverse_mapping
            
        return [reverse_mapping[idx] for idx in indices]

    def calculate_ac_value(self, numbers: List[int]) -> int:
        """计算AC值"""
        if len(numbers) < 2:
            return 0

        diffs = set()
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                diffs.add(abs(numbers[i] - numbers[j]))

        return len(diffs) - (len(numbers) - 1)

    def is_valid_ac_value(self, numbers: List[int], ball_type: str) -> bool:
        """检查AC值是否有效"""
        ac_value = self.calculate_ac_value(numbers)
        config = self.ac_config[ball_type]
        return config["min_ac"] <= ac_value <= config["max_ac"]

    def analyze_zone_distribution(self, numbers: List[int], ball_type: str) -> Dict:
        """分析号码的区间分布"""
        zone_data = self.zone_config[f"{ball_type}_zones"]
        zone_stats = {zone: 0 for zone in zone_data.keys()}

        for num in numbers:
            for zone_name, zone_nums in zone_data.items():
                if num in zone_nums:
                    zone_stats[zone_name] += 1
                    break

        return zone_stats

    def calculate_zone_score(self, numbers: List[int], ball_type: str) -> float:
        """计算断区得分（0-1之间）"""
        zone_stats = self.analyze_zone_distribution(numbers, ball_type)
        total_zones = len(zone_stats)
        covered_zones = sum(1 for count in zone_stats.values() if count > 0)

        return covered_zones / total_zones

    def is_number_in_specified_set(self, numbers: List[int], ball_type: str) -> bool:
        """检查号码是否在指定集合中"""
        if ball_type == "red":
            specified_set = set(self.specified_red_numbers)
        else:
            specified_set = set(self.specified_blue_numbers)
            
        return all(num in specified_set for num in numbers)

    def genetic_algorithm_enhancement(self, x_data: np.ndarray, y_data: np.ndarray,
                                      ball_type: str, num_augment: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """遗传算法数据增强（修复维度问题）"""
        enhanced_x, enhanced_y = [], []

        # 记录原始数据维度
        original_x_dims = x_data.shape
        logger.info(f"原始x_data维度: {original_x_dims}")

        for _ in range(num_augment):
            # 随机选择父代样本
            parent_indices = np.random.choice(len(x_data), size=2, replace=False)
            parent1_x, parent1_y = x_data[parent_indices[0]], y_data[parent_indices[0]]
            parent2_x, parent2_y = x_data[parent_indices[1]], y_data[parent_indices[1]]

            # 交叉操作
            child_x, child_y = self._crossover(parent1_x, parent1_y, parent2_x, parent2_y, ball_type)

            # 变异操作
            child_x, child_y = self._mutate(child_x, child_y, ball_type)

            # 只保留有效的增强样本（在指定集合中）
            child_y_numbers = self.map_from_specified_index(child_y.tolist(), ball_type)
            if self._is_valid_augmented_sample(np.array(child_y_numbers), ball_type):
                # 确保增强数据维度与原始数据一致
                if len(child_x.shape) != len(original_x_dims):
                    # 如果维度不一致，尝试reshape
                    try:
                        child_x = child_x.reshape(original_x_dims[1:])
                    except Exception as e:
                        logger.warning(f"无法reshape增强数据: {e}")
                        continue

                enhanced_x.append(child_x)
                enhanced_y.append(child_y)

        if enhanced_x:
            enhanced_x = np.array(enhanced_x)
            enhanced_y = np.array(enhanced_y)
            logger.info(f"增强数据维度 - x: {enhanced_x.shape}, y: {enhanced_y.shape}")
        else:
            enhanced_x = np.empty((0, *original_x_dims[1:]))
            enhanced_y = np.empty((0, *y_data.shape[1:]))
            logger.warning("未生成有效的增强数据")

        return enhanced_x, enhanced_y

    def _crossover(self, parent1_x, parent1_y, parent2_x, parent2_y, ball_type: str):
        """交叉操作（修复维度处理）"""
        # 根据数据维度进行不同的交叉策略
        if len(parent1_x.shape) == 3:
            # 3维数据交叉 (样本, 时间步, 特征)
            crossover_point = random.randint(1, parent1_x.shape[1] - 1)
            child_x = np.concatenate([
                parent1_x[:, :crossover_point, :],
                parent2_x[:, crossover_point:, :]
            ], axis=1)
        elif len(parent1_x.shape) == 2:
            # 2维数据交叉 (样本, 特征)
            crossover_point = random.randint(1, parent1_x.shape[1] - 1)
            child_x = np.concatenate([
                parent1_x[:, :crossover_point],
                parent2_x[:, crossover_point:]
            ], axis=1)
        else:
            # 1维数据交叉
            crossover_point = random.randint(1, len(parent1_x) - 1)
            child_x = np.concatenate([parent1_x[:crossover_point], parent2_x[crossover_point:]])

        # 目标交叉（基于概率混合）
        if ball_type == "red" or (ball_type == "blue" and self.name == "dlt"):
            # 多号码交叉
            if len(parent1_y.shape) > 1:
                # 多维标签交叉
                crossover_mask = np.random.random(parent1_y.shape) > 0.5
                child_y = np.where(crossover_mask, parent1_y, parent2_y)
            else:
                # 一维标签交叉
                crossover_mask = np.random.random(len(parent1_y)) > 0.5
                child_y = np.where(crossover_mask, parent1_y, parent2_y)
        else:
            # 单号码交叉（50%概率选择父代）
            child_y = parent1_y if random.random() > 0.5 else parent2_y

        return child_x, child_y

    def _mutate(self, x_sequence, y_target, ball_type: str):
        """变异操作（修复维度处理）"""
        # 序列变异
        mutated_x = x_sequence.copy()

        # 根据数据维度进行变异
        if len(mutated_x.shape) == 3:
            # 3维数据变异
            for i in range(mutated_x.shape[0]):
                for j in range(mutated_x.shape[1]):
                    for k in range(mutated_x.shape[2]):
                        if random.random() < 0.1:  # 10%的变异概率
                            mutation_strength = random.randint(-2, 2)
                            # 使用指定号码范围
                            max_idx = len(self.specified_red_numbers) - 1 if ball_type == "red" else len(self.specified_blue_numbers) - 1
                            mutated_x[i, j, k] = np.clip(mutated_x[i, j, k] + mutation_strength, 0, max_idx)
        elif len(mutated_x.shape) == 2:
            # 2维数据变异
            for i in range(mutated_x.shape[0]):
                for j in range(mutated_x.shape[1]):
                    if random.random() < 0.1:  # 10%的变异概率
                        mutation_strength = random.randint(-2, 2)
                        max_idx = len(self.specified_red_numbers) - 1 if ball_type == "red" else len(self.specified_blue_numbers) - 1
                        mutated_x[i, j] = np.clip(mutated_x[i, j] + mutation_strength, 0, max_idx)
        else:
            # 1维数据变异
            for i in range(len(mutated_x)):
                if random.random() < 0.1:  # 10%的变异概率
                    mutation_strength = random.randint(-2, 2)
                    max_idx = len(self.specified_red_numbers) - 1 if ball_type == "red" else len(self.specified_blue_numbers) - 1
                    mutated_x[i] = np.clip(mutated_x[i] + mutation_strength, 0, max_idx)

        # 目标变异
        mutated_y = y_target.copy()
        if ball_type == "red" or (ball_type == "blue" and self.name == "dlt"):
            # 多号码变异
            if len(mutated_y.shape) > 1:
                for i in range(mutated_y.shape[0]):
                    for j in range(mutated_y.shape[1]):
                        if random.random() < 0.15:  # 15%的变异概率
                            mutation_strength = random.randint(-3, 3)
                            max_idx = len(self.specified_red_numbers) - 1 if ball_type == "red" else len(self.specified_blue_numbers) - 1
                            mutated_y[i, j] = np.clip(mutated_y[i, j] + mutation_strength, 0, max_idx)
            else:
                for i in range(len(mutated_y)):
                    if random.random() < 0.15:  # 15%的变异概率
                        mutation_strength = random.randint(-3, 3)
                        max_idx = len(self.specified_red_numbers) - 1 if ball_type == "red" else len(self.specified_blue_numbers) - 1
                        mutated_y[i] = np.clip(mutated_y[i] + mutation_strength, 0, max_idx)
        else:
            # 单号码变异
            if random.random() < 0.2:  # 20%的变异概率
                mutation_strength = random.randint(-3, 3)
                max_idx = len(self.specified_blue_numbers) - 1
                mutated_y = np.clip(mutated_y + mutation_strength, 0, max_idx)

        return mutated_x, mutated_y

    def _get_max_number(self, ball_type: str) -> int:
        """获取最大号码值"""
        if ball_type == "red":
            return max(self.specified_red_numbers)
        else:
            return max(self.specified_blue_numbers)

    def _is_valid_augmented_sample(self, numbers: np.ndarray, ball_type: str) -> bool:
        """检查增强样本是否有效"""
        numbers_list = numbers.tolist()
        
        # 检查号码是否在指定集合中
        if not self.is_number_in_specified_set(numbers_list, ball_type):
            logger.warning(f"增强样本包含非指定号码: {numbers_list}")
            return False
        
        # 检查重复号码（多号码情况）
        if len(numbers_list) > 1 and len(set(numbers_list)) != len(numbers_list):
            logger.warning(f"增强样本包含重复号码: {numbers_list}")
            return False
        
        # 检查AC值
        if not self.is_valid_ac_value(numbers_list, ball_type):
            logger.warning(f"增强样本AC值无效: {numbers_list}, AC={self.calculate_ac_value(numbers_list)}")
            return False
        
        return True


def save_summary_to_file(name, summary_content):
    """将总结内容保存到文件"""
    current_date = datetime.now().strftime("%Y%m%d")
    filename = f"{name}_training_summary_{current_date}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    logger.info(f"训练总结已保存到文件: {filename}")


def map_data_to_indices(data_array, enhancer, ball_type="red"):
    """将数据中的实际号码映射到指定号码的索引"""
    mapped_data = []
    for sample in data_array:
        if len(sample.shape) > 1:
            # 多维度数据 (windows_size, num_balls)
            mapped_sample = []
            for seq in sample:
                mapped_seq = []
                for num in seq:
                    if ball_type == "red":
                        specified_set = enhancer.specified_red_numbers
                    else:
                        specified_set = enhancer.specified_blue_numbers
                    
                    if num in specified_set:
                        idx = enhancer.map_to_specified_index([num], ball_type)[0]
                        mapped_seq.append(idx)
                    else:
                        # 如果不在指定集合中，找到最接近的指定号码
                        closest_num = min(specified_set, key=lambda x: abs(x - num))
                        idx = enhancer.map_to_specified_index([closest_num], ball_type)[0]
                        mapped_seq.append(idx)
                        logger.warning(f"号码 {num} 不在{ball_type}球指定集合中，使用最接近的 {closest_num}")
                mapped_sample.append(mapped_seq)
            mapped_data.append(mapped_sample)
        else:
            # 单维度数据
            mapped_sample = []
            for num in sample:
                if ball_type == "red":
                    specified_set = enhancer.specified_red_numbers
                else:
                    specified_set = enhancer.specified_blue_numbers
                
                if num in specified_set:
                    idx = enhancer.map_to_specified_index([num], ball_type)[0]
                    mapped_sample.append(idx)
                else:
                    closest_num = min(specified_set, key=lambda x: abs(x - num))
                    idx = enhancer.map_to_specified_index([closest_num], ball_type)[0]
                    mapped_sample.append(idx)
                    logger.warning(f"号码 {num} 不在{ball_type}球指定集合中，使用最接近的 {closest_num}")
            mapped_data.append(mapped_sample)
    return np.array(mapped_data)


def create_data(data, name, windows):
    """创建训练数据 - 确保大乐透蓝球数据正确分割，使用指定号码"""
    if not len(data):
        raise logger.error("请执行 get_data.py 进行数据下载！")
    else:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logger.info("训练数据已加载!")

    data = data.iloc[:, 2:].values
    logger.info("原始训练集数据维度: {}".format(data.shape))
    
    # 创建增强器实例
    enhancer = AlgorithmEnhancer(name)
    
    # 过滤数据：只保留号码都在指定集合中的行
    filtered_data = []
    logger.info("开始过滤数据，只保留指定号码...")
    
    for i, row in enumerate(data):
        if name == "ssq":
            red_balls = row[:6].tolist()
            blue_balls = row[6:7].tolist()
        else:
            red_balls = row[:5].tolist()
            blue_balls = row[5:7].tolist()
        
        # 检查号码是否都在指定集合中
        red_valid = enhancer.is_number_in_specified_set(red_balls, "red")
        blue_valid = enhancer.is_number_in_specified_set(blue_balls, "blue")
        
        if red_valid and blue_valid:
            filtered_data.append(row)
    
    data = np.array(filtered_data)
    logger.info(f"过滤后数据维度: {data.shape}")
    logger.info(f"保留了 {len(data)}/{len(data)} 期有效数据")
    
    x_data, y_data = [], []
    for i in range(len(data) - windows - 1):
        sub_data = data[i:(i + windows + 1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])

    # 关键修复：正确分割大乐透红球和蓝球
    if name == "ssq":
        # 双色球：6个红球 + 1个蓝球
        red_cut_num = 6
        blue_cut_num = 1
    else:
        # 大乐透：5个红球 + 2个蓝球
        red_cut_num = 5
        blue_cut_num = 2

    # 映射数据到指定号码索引
    logger.info("将数据映射到指定号码索引...")
    
    # 红球数据
    x_red_data = np.array(x_data)[:, :, :red_cut_num]
    y_red_data = np.array(y_data)[:, :red_cut_num]
    
    x_red_mapped = map_data_to_indices(x_red_data, enhancer, "red")
    y_red_mapped = map_data_to_indices(y_red_data, enhancer, "red")
    
    # 蓝球数据
    x_blue_data = np.array(x_data)[:, :, red_cut_num:red_cut_num + blue_cut_num]
    y_blue_data = np.array(y_data)[:, red_cut_num:red_cut_num + blue_cut_num]
    
    x_blue_mapped = map_data_to_indices(x_blue_data, enhancer, "blue")
    y_blue_mapped = map_data_to_indices(y_blue_data, enhancer, "blue")
    
    logger.info(f"红球数据维度 - x: {x_red_mapped.shape}, y: {y_red_mapped.shape}")
    logger.info(f"蓝球数据维度 - x: {x_blue_mapped.shape}, y: {y_blue_mapped.shape}")

    return {
        "red": {
            "x_data": x_red_mapped,
            "y_data": y_red_mapped
        },
        "blue": {
            "x_data": x_blue_mapped,
            "y_data": y_blue_mapped
        }
    }


def create_train_test_data(name, windows, train_test_split):
    """划分数据集"""
    if train_test_split < 0.5:
        raise Exception("训练集采样比例小于50%,训练终止,请求重新采样（train_test_split>0.5）!")

    path = "{}{}".format(name_path[name]["path"], data_file_name)
    data = pd.read_csv(path)
    logger.info("read data from path: {}".format(path))
    logger.info(f"原始数据期数: {len(data)}")

    # 创建增强器用于日志
    enhancer = AlgorithmEnhancer(name)
    logger.info(f"使用指定号码 - 红球: {enhancer.specified_red_numbers}")
    logger.info(f"使用指定号码 - 蓝球: {enhancer.specified_blue_numbers}")

    train_data = create_data(data.iloc[:int(len(data) * train_test_split)], name, windows)
    test_data = create_data(data.iloc[int(len(data) * train_test_split):], name, windows)

    logger.info("train_data sample rate = {}, test_data sample rate = {}".format(
        train_test_split, round(1 - train_test_split, 2)))

    return train_data, test_data


def get_topk_predictions(probabilities, topk, n_class, enhancer=None, ball_type="red"):
    """获取概率最高的topk个预测号码 - 修复版本，返回指定号码"""
    try:
        # 处理不同维度的概率输出
        if probabilities.ndim == 3:
            # 3D输出 (batch, sequence, class) - 对序列维度求平均
            avg_probs = np.mean(probabilities, axis=1)[0]  # 取第一个batch
        elif probabilities.ndim == 2:
            # 2D输出 (batch, class)
            avg_probs = probabilities[0]
        elif probabilities.ndim == 1:
            # 1D输出 (class,)
            avg_probs = probabilities
        else:
            avg_probs = np.ones(n_class) / n_class

        # 确保概率数组长度正确
        if len(avg_probs) > n_class:
            avg_probs = avg_probs[:n_class]
        elif len(avg_probs) < n_class:
            # 填充缺失的概率值
            padded_probs = np.ones(n_class) * 0.001  # 小概率值
            padded_probs[:len(avg_probs)] = avg_probs
            avg_probs = padded_probs

        # 归一化
        if np.sum(avg_probs) > 0:
            avg_probs = avg_probs / np.sum(avg_probs)
        else:
            avg_probs = np.ones(n_class) / n_class

        # 获取topk预测
        topk_indices = np.argsort(avg_probs)[-topk:][::-1]
        
        # 如果有enhancer，映射回指定号码
        if enhancer:
            if ball_type == "red":
                topk_numbers = [enhancer.red_reverse_mapping[idx] for idx in topk_indices]
            else:
                topk_numbers = [enhancer.blue_reverse_mapping[idx] for idx in topk_indices]
        else:
            topk_numbers = [idx + 1 for idx in topk_indices]
            
        topk_probs = [avg_probs[idx] for idx in topk_indices]

        return topk_numbers, topk_probs

    except Exception as e:
        logger.error(f"获取topk预测时出错: {e}")
        # 返回指定号码中的前topk个作为默认预测
        if ball_type == "red":
            default_numbers = SPECIFIED_RED_NUMBERS[:min(topk, len(SPECIFIED_RED_NUMBERS))]
        else:
            default_numbers = SPECIFIED_BLUE_NUMBERS[:min(topk, len(SPECIFIED_BLUE_NUMBERS))]
            
        default_probs = [1.0 / len(default_numbers)] * len(default_numbers)
        return default_numbers, default_probs



def train_with_eval_red_ball_model(name, x_train, y_train, x_test, y_test):
    """红球模型训练与评估 - 集成算法增强，使用指定号码"""
    m_args = model_args[name]

    # 创建算法增强器
    enhancer = AlgorithmEnhancer(name)
    
    logger.info(f"红球训练使用指定号码: {enhancer.specified_red_numbers}")

    # 应用算法增强
    if m_args["model_args"]["enable_algorithm"]:
        logger.info("应用算法增强到红球训练数据...")
        augmented_x, augmented_y = enhancer.genetic_algorithm_enhancement(x_train, y_train, "red", num_augment=20)

        # 合并原始数据和增强数据
        x_train = np.concatenate([x_train, augmented_x], axis=0)
        y_train = np.concatenate([y_train, augmented_y], axis=0)
        logger.info(f"增强后训练数据: {x_train.shape}, 增强后标签: {y_train.shape}")

    # 修复：检查数据范围
    logger.info(f"数据范围检查 - x_train最小: {np.min(x_train)}, 最大: {np.max(x_train)}")
    logger.info(f"数据范围检查 - y_train最小: {np.min(y_train)}, 最大: {np.max(y_train)}")

    # 数据已经是索引形式，不需要减1
    train_data_len = x_train.shape[0]
    logger.info("训练特征数据维度: {}".format(x_train.shape))
    logger.info("训练标签数据维度: {}".format(y_train.shape))

    # 同样处理测试数据
    test_data_len = x_test.shape[0]
    logger.info("测试特征数据维度: {}".format(x_test.shape))
    logger.info("测试标签数据维度: {}".format(y_test.shape))

    red_topk = m_args["model_args"]["red_output_count"]
    red_n_class = m_args["model_args"]["red_n_class"]

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        # 使用增强模型
        red_ball_model = EnhancedLstmWithCRFModel(
            batch_size=m_args["model_args"]["batch_size"],
            n_class=m_args["model_args"]["red_n_class"],
            ball_num=m_args["model_args"]["sequence_len"] if name == "ssq" else m_args["model_args"][
                "red_sequence_len"],
            w_size=m_args["model_args"]["windows_size"],
            embedding_size=m_args["model_args"]["red_embedding_size"],
            words_size=m_args["model_args"]["red_n_class"],
            hidden_size=m_args["model_args"]["red_hidden_size"],
            layer_size=m_args["model_args"]["red_layer_size"],
            name=name,
            ball_type="red"
        )

        # 使用增强损失函数
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["red_learning_rate"],
            beta1=m_args["train_args"]["red_beta1"],
            beta2=m_args["train_args"]["red_beta2"],
            epsilon=m_args["train_args"]["red_epsilon"]
        ).minimize(red_ball_model.enhanced_loss)

        sess.run(tf.compat.v1.global_variables_initializer())

        sequence_len = m_args["model_args"]["sequence_len"] if name == "ssq" else m_args["model_args"][
            "red_sequence_len"]

        for epoch in range(m_args["model_args"]["red_epochs"]):
            epoch_loss = 0
            for i in range(train_data_len):
                # 修复：添加数据范围检查
                current_x = x_train[i:(i + 1), :, :]
                current_y = y_train[i:(i + 1), :]

                # 检查数据是否有效
                if np.any(current_x < 0) or np.any(current_y < 0):
                    logger.warning(
                        f"发现无效数据在epoch {epoch}, 样本 {i}: x_min={np.min(current_x)}, y_min={np.min(current_y)}")
                    # 跳过无效数据
                    continue

                try:
                    # 关键修复：使用模型属性而不是硬编码的字符串
                    _, loss_, pred, probs = sess.run([
                        train_step, red_ball_model.enhanced_loss, red_ball_model.pred_sequence,
                        red_ball_model.probabilities
                    ], feed_dict={
                        red_ball_model.inputs: current_x,  # 使用模型属性
                        red_ball_model.tag_indices: current_y,  # 使用模型属性
                        red_ball_model.sequence_length: np.array([sequence_len] * 1)  # 使用模型属性
                    })

                    epoch_loss += loss_

                    # 获取复式预测号码（映射回指定号码）
                    topk_numbers, topk_probs = get_topk_predictions(probs, red_topk, red_n_class, enhancer, "red")

                    if i % 100 == 0:
                        # 将索引映射回实际号码
                        true_numbers = enhancer.map_from_specified_index(current_y[0].tolist(), "red")
                        pred_numbers = enhancer.map_from_specified_index(pred[0].tolist(), "red")

                        # 计算算法指标
                        true_ac = enhancer.calculate_ac_value(true_numbers)
                        pred_ac = enhancer.calculate_ac_value(pred_numbers)
                        zone_score = enhancer.calculate_zone_score(pred_numbers, "red")

                        logger.info("epoch: {}, loss: {:.4f}".format(epoch, loss_))
                        logger.info("真实号码: {} (AC值: {})".format(true_numbers, true_ac))
                        logger.info("CRF预测号码: {} (AC值: {})".format(pred_numbers, pred_ac))
                        logger.info("复式预测号码 (Top{}): {}".format(red_topk, topk_numbers))
                        logger.info("断区得分: {:.3f}".format(zone_score))
                        logger.info("---")

                except Exception as e:
                    logger.error(f"训练过程中出错 (epoch {epoch}, sample {i}): {e}")
                    continue

            # 每个epoch记录平均损失
            if train_data_len > 0:
                avg_epoch_loss = epoch_loss / train_data_len
                logger.info("Epoch {} 平均损失: {:.4f}".format(epoch, avg_epoch_loss))

        logger.info("训练耗时: {}".format(time.time() - start_time))

        # 关键修改：保存节点名称时添加前缀
        pred_key[f"red_{ball_name[0][0]}"] = red_ball_model.pred_sequence.name
        pred_key[f"red_{ball_name[0][0]}_logits"] = red_ball_model.outputs.name

        if not os.path.exists(m_args["path"]["red"]):
            os.makedirs(m_args["path"]["red"])

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["red"], red_ball_model_name, extension))

        # 模型评估代码...
        logger.info("红球模型评估【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        eval_topk_d = {}
        eval_ac_d = {}  # AC值命中统计
        eval_zone_d = {}  # 断区得分统计

        for j in range(test_data_len):
            try:
                true = y_test[j:(j + 1), :]

                # 关键修复：评估时也要使用正确的feed_dict
                pred, probs = sess.run([
                    red_ball_model.pred_sequence, red_ball_model.probabilities
                ], feed_dict={
                    red_ball_model.inputs: x_test[j:(j + 1), :, :],  # 使用模型属性
                    red_ball_model.sequence_length: np.array([sequence_len] * 1)  # 使用模型属性
                })

                # 计算CRF预测的命中数（映射回指定号码）
                true_numbers = enhancer.map_from_specified_index(true[0].tolist(), "red")
                pred_numbers = enhancer.map_from_specified_index(pred[0].tolist(), "red")
                count = len(set(true_numbers) & set(pred_numbers))
                all_true_count += count

                # 计算复式预测的命中数
                topk_numbers, _ = get_topk_predictions(probs, red_topk, red_n_class, enhancer, "red")
                topk_hit_count = 0
                for num in true_numbers:
                    if num in topk_numbers:
                        topk_hit_count += 1

                # 计算AC值指标
                true_ac = enhancer.calculate_ac_value(true_numbers)
                pred_ac = enhancer.calculate_ac_value(pred_numbers)
                ac_diff = abs(true_ac - pred_ac)

                # 计算断区指标
                zone_score = enhancer.calculate_zone_score(pred_numbers, "red")

                if count in eval_d:
                    eval_d[count] += 1
                else:
                    eval_d[count] = 1

                if topk_hit_count in eval_topk_d:
                    eval_topk_d[topk_hit_count] += 1
                else:
                    eval_topk_d[topk_hit_count] = 1

                if ac_diff in eval_ac_d:
                    eval_ac_d[ac_diff] += 1
                else:
                    eval_ac_d[ac_diff] = 1

                zone_score_key = round(zone_score, 1)
                if zone_score_key in eval_zone_d:
                    eval_zone_d[zone_score_key] += 1
                else:
                    eval_zone_d[zone_score_key] = 1

            except Exception as e:
                logger.error(f"评估过程中出错 (sample {j}): {e}")
                continue

        logger.info("测试期数: {}".format(test_data_len))

        # 构建红球总结内容
        red_summary = []
        red_summary.append("红球模型评估结果")
        red_summary.append("=" * 50)
        red_summary.append(f"测试期数: {test_data_len}")
        red_summary.append(f"使用指定号码: {enhancer.specified_red_numbers}")

        # 输出CRF预测结果
        red_summary.append("CRF预测结果:")
        for k, v in sorted(eval_d.items()):
            red_summary.append("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        # 输出复式预测结果
        red_summary.append("复式预测结果 (Top{}):".format(red_topk))
        for k, v in sorted(eval_topk_d.items()):
            red_summary.append("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        # 输出AC值指标
        red_summary.append("AC值差异分布:")
        for k, v in sorted(eval_ac_d.items()):
            red_summary.append("AC值差异{}，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        # 输出断区指标
        red_summary.append("断区得分分布:")
        for k, v in sorted(eval_zone_d.items()):
            red_summary.append("断区得分{}，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        red_summary.append("整体准确率: {}%".format(
            round(all_true_count * 100 / (test_data_len * sequence_len), 2)
        ))
        red_summary.append("")

        # 将红球总结内容输出到日志
        for line in red_summary:
            logger.info(line)

        return "\n".join(red_summary)


def train_with_eval_blue_ball_model(name, x_train, y_train, x_test, y_test):
    """蓝球模型训练与评估 - 修复大乐透蓝球维度问题，使用指定号码"""
    m_args = model_args[name]

    # 关键修复：确保大乐透蓝球使用正确的输入维度
    if name == "dlt":
        # 大乐透蓝球应该是2个号码，检查并修正数据维度
        expected_features = 2
        actual_features = x_train.shape[2] if len(x_train.shape) > 2 else 1

        logger.info(f"大乐透蓝球数据维度检查: 期望={expected_features}, 实际={actual_features}")

        if actual_features != expected_features:
            logger.warning(f"数据维度不匹配! 将进行修正")
            # 重新reshape数据为正确的维度
            if actual_features > expected_features:
                # 如果特征数过多，只取前2个特征
                x_train = x_train[:, :, :expected_features]
                y_train = y_train[:, :expected_features]
                logger.info(f"修正后训练数据形状: x_train={x_train.shape}, y_train={y_train.shape}")
            elif actual_features < expected_features:
                # 如果特征数不足，无法修复，报错
                raise ValueError(f"蓝球数据特征数不足: 需要{expected_features}个，只有{actual_features}个")

    # 创建算法增强器
    enhancer = AlgorithmEnhancer(name)
    
    logger.info(f"蓝球训练使用指定号码: {enhancer.specified_blue_numbers}")

    # 应用算法增强
    if m_args["model_args"]["enable_algorithm"]:
        logger.info("应用算法增强到蓝球训练数据...")
        try:
            augmented_x, augmented_y = enhancer.genetic_algorithm_enhancement(
                x_train, y_train, "blue", num_augment=15
            )
            if len(augmented_x) > 0:
                x_train = np.concatenate([x_train, augmented_x], axis=0)
                y_train = np.concatenate([y_train, augmented_y], axis=0)
                logger.info(f"增强后训练数据: {x_train.shape}, 增强后标签: {y_train.shape}")
        except Exception as e:
            logger.error(f"算法增强过程中出错: {e}")

    # 数据已经是索引形式，不需要减1
    train_data_len = x_train.shape[0]
    test_data_len = x_test.shape[0]

    logger.info("处理后的训练特征数据维度: {}".format(x_train.shape))
    logger.info("处理后的训练标签数据维度: {}".format(y_train.shape))

    blue_topk = m_args["model_args"]["blue_output_count"]
    blue_n_class = m_args["model_args"]["blue_n_class"]
    blue_epochs = m_args["model_args"]["blue_epochs"]

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        if name == "ssq":
            # 双色球蓝球模型（单号码）- 使用SignalLSTM
            blue_ball_model = EnhancedSignalLstmModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                outputs_size=m_args["model_args"]["blue_n_class"],
                layer_size=m_args["model_args"]["blue_layer_size"],
                name=name,
                ball_type="blue"
            )

            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=m_args["train_args"]["blue_learning_rate"],
                beta1=m_args["train_args"]["blue_beta1"],
                beta2=m_args["train_args"]["blue_beta2"],
                epsilon=m_args["train_args"]["blue_epsilon"]
            ).minimize(blue_ball_model.enhanced_loss)

            sess.run(tf.compat.v1.global_variables_initializer())

            # 训练过程
            for epoch in range(blue_epochs):
                epoch_loss = 0
                valid_samples = 0

                for i in range(train_data_len):
                    try:
                        current_x = x_train[i:(i + 1), :]
                        current_y = y_train[i:(i + 1)]

                        # 跳过无效数据
                        if np.any(current_x < 0) or np.any(current_y < 0):
                            continue

                        # 训练步骤 - 关键修复：使用模型属性
                        _, loss_, pred, probs = sess.run([
                            train_step, blue_ball_model.enhanced_loss,
                            blue_ball_model.pred_label, blue_ball_model.probabilities
                        ], feed_dict={
                            blue_ball_model.inputs: current_x,
                            blue_ball_model.tag_indices: tf.keras.utils.to_categorical(current_y,
                                                                                       num_classes=blue_n_class)
                        })

                        epoch_loss += loss_
                        valid_samples += 1

                        if i % 100 == 0:
                            true_number = enhancer.map_from_specified_index([current_y[0]], "blue")[0]
                            pred_number = enhancer.map_from_specified_index([pred[0]], "blue")[0]
                            logger.info("epoch: {}, loss: {:.4f}".format(epoch, loss_))
                            logger.info("真实号码: {}, 预测号码: {}".format(true_number, pred_number))

                    except Exception as e:
                        logger.error(f"训练过程中出错 (epoch {epoch}, sample {i}): {e}")
                        continue

                if valid_samples > 0:
                    avg_epoch_loss = epoch_loss / valid_samples
                    logger.info("Epoch {} 平均损失: {:.4f}".format(epoch, avg_epoch_loss))

        else:
            # 大乐透蓝球模型（2个号码）- 使用CRF模型
            blue_sequence_len = m_args["model_args"]["blue_sequence_len"]

            logger.info(f"大乐透蓝球配置: 序列长度={blue_sequence_len}, 窗口大小={m_args['model_args']['windows_size']}")
            logger.info(f"实际数据维度: x_train={x_train.shape}, y_train={y_train.shape}")

            # 确保数据维度正确
            if x_train.shape[2] != blue_sequence_len:
                if x_train.shape[2] > blue_sequence_len:
                    x_train = x_train[:, :, :blue_sequence_len]
                    y_train = y_train[:, :blue_sequence_len]
                elif x_train.shape[2] < blue_sequence_len:
                    pad_width = blue_sequence_len - x_train.shape[2]
                    x_train = np.pad(x_train, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                    y_train = np.pad(y_train, ((0, 0), (0, pad_width)), mode='constant')

            # 同样修正测试数据
            if x_test.shape[2] != blue_sequence_len:
                if x_test.shape[2] > blue_sequence_len:
                    x_test = x_test[:, :, :blue_sequence_len]
                    y_test = y_test[:, :blue_sequence_len]
                elif x_test.shape[2] < blue_sequence_len:
                    pad_width = blue_sequence_len - x_test.shape[2]
                    x_test = np.pad(x_test, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                    y_test = np.pad(y_test, ((0, 0), (0, pad_width)), mode='constant')

            blue_ball_model = EnhancedLstmWithCRFModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                ball_num=blue_sequence_len,
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                words_size=m_args["model_args"]["blue_n_class"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                layer_size=m_args["model_args"]["blue_layer_size"],
                name=name,
                ball_type="blue"
            )

            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=m_args["train_args"]["blue_learning_rate"],
                beta1=m_args["train_args"]["blue_beta1"],
                beta2=m_args["train_args"]["blue_beta2"],
                epsilon=m_args["train_args"]["blue_epsilon"]
            ).minimize(blue_ball_model.enhanced_loss)

            sess.run(tf.compat.v1.global_variables_initializer())

            for epoch in range(blue_epochs):
                epoch_loss = 0
                valid_samples = 0

                for i in range(train_data_len):
                    try:
                        current_x = x_train[i:(i + 1), :, :]
                        current_y = y_train[i:(i + 1), :]

                        # 跳过无效数据
                        if np.any(current_x < 0) or np.any(current_y < 0):
                            continue

                        # 训练步骤 - 关键修复：使用模型属性
                        _, loss_, pred, probs = sess.run([
                            train_step, blue_ball_model.enhanced_loss,
                            blue_ball_model.pred_sequence, blue_ball_model.probabilities
                        ], feed_dict={
                            blue_ball_model.inputs: current_x,
                            blue_ball_model.tag_indices: current_y,
                            blue_ball_model.sequence_length: np.array([blue_sequence_len] * 1)
                        })

                        epoch_loss += loss_
                        valid_samples += 1

                        if i % 100 == 0:
                            true_numbers = enhancer.map_from_specified_index(current_y[0].tolist(), "blue")
                            pred_numbers = enhancer.map_from_specified_index(pred[0].tolist(), "blue")

                            # 获取复式预测（Top 5）
                            topk_numbers, topk_probs = get_topk_predictions(probs, blue_topk, blue_n_class, enhancer, "blue")

                            logger.info("epoch: {}, loss: {:.4f}".format(epoch, loss_))
                            logger.info("真实号码: {}, CRF预测号码: {}".format(true_numbers, pred_numbers))
                            logger.info("复式预测Top{}: {}".format(blue_topk, topk_numbers))

                    except Exception as e:
                        logger.error(f"训练过程中出错 (epoch {epoch}, sample {i}): {e}")
                        continue

                if valid_samples > 0:
                    avg_epoch_loss = epoch_loss / valid_samples
                    logger.info("Epoch {} 平均损失: {:.4f}, 有效样本数: {}".format(epoch, avg_epoch_loss, valid_samples))

        logger.info("训练耗时: {:.2f}秒".format(time.time() - start_time))

        # 关键修改：保存节点名称时添加完整的作用域前缀
        scope_name = f"{name}_blue_model"
        if name == "ssq":
            pred_key[f"blue_{ball_name[1][0]}"] = f"{scope_name}/{blue_ball_model.pred_label.name}"
            pred_key[f"blue_{ball_name[1][0]}_logits"] = f"{scope_name}/{blue_ball_model.outputs.name}"
        else:
            pred_key[f"blue_{ball_name[1][0]}"] = f"{scope_name}/{blue_ball_model.pred_sequence.name}"
            pred_key[f"blue_{ball_name[1][0]}_logits"] = f"{scope_name}/{blue_ball_model.outputs.name}"

        if not os.path.exists(m_args["path"]["blue"]):
            os.makedirs(m_args["path"]["blue"])

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["blue"], blue_ball_model_name, extension))

        # 模型评估
        logger.info("蓝球模型评估【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        eval_topk_d = {}
        all_true_count = 0

        sequence_len = 1 if name == "ssq" else blue_sequence_len

        for j in range(test_data_len):
            try:
                if name == "ssq":
                    # 双色球评估
                    true = y_test[j:(j + 1)]

                    # 关键修复：评估时使用正确的feed_dict
                    pred, probs = sess.run([
                        blue_ball_model.pred_label, blue_ball_model.probabilities
                    ], feed_dict={
                        blue_ball_model.inputs: x_test[j:(j + 1), :]
                    })

                    true_number = enhancer.map_from_specified_index([true[0]], "blue")[0]
                    pred_number = enhancer.map_from_specified_index([pred[0]], "blue")[0]
                    count = 1 if true_number == pred_number else 0
                    all_true_count += count

                    # 复式预测
                    topk_numbers, topk_probs = get_topk_predictions(probs, blue_topk, blue_n_class, enhancer, "blue")
                    topk_hit_count = 1 if true_number in topk_numbers else 0

                else:
                    # 大乐透评估
                    true = y_test[j:(j + 1), :]

                    # 关键修复：评估时使用正确的feed_dict
                    pred, probs = sess.run([
                        blue_ball_model.pred_sequence, blue_ball_model.probabilities
                    ], feed_dict={
                        blue_ball_model.inputs: x_test[j:(j + 1), :, :],
                        blue_ball_model.sequence_length: np.array([blue_sequence_len] * 1)
                    })

                    true_numbers = enhancer.map_from_specified_index(true[0].tolist(), "blue")
                    pred_numbers = enhancer.map_from_specified_index(pred[0].tolist(), "blue")

                    # 计算CRF预测命中数
                    count = len(set(true_numbers) & set(pred_numbers))
                    all_true_count += count

                    # 复式预测（Top 5）
                    topk_numbers, topk_probs = get_topk_predictions(probs, blue_topk, blue_n_class, enhancer, "blue")
                    topk_hit_count = 0
                    for num in true_numbers:
                        if num in topk_numbers:
                            topk_hit_count += 1

                # 统计结果
                eval_d[count] = eval_d.get(count, 0) + 1
                eval_topk_d[topk_hit_count] = eval_topk_d.get(topk_hit_count, 0) + 1

            except Exception as e:
                logger.error(f"评估过程中出错 (sample {j}): {e}")
                continue

        # 输出评估结果
        logger.info("测试期数: {}".format(test_data_len))

        # CRF预测结果
        logger.info("CRF预测结果:")
        for k, v in sorted(eval_d.items()):
            percentage = round(v * 100 / test_data_len, 2)
            logger.info("命中{}个球，{}期，占比: {}%".format(k, v, percentage))

        # 复式预测结果
        logger.info("复式预测结果 (Top{}):".format(blue_topk))
        for k, v in sorted(eval_topk_d.items()):
            percentage = round(v * 100 / test_data_len, 2)
            logger.info("命中{}个球，{}期，占比: {}%".format(k, v, percentage))

        # 计算准确率
        if name == "ssq":
            accuracy = round(all_true_count * 100 / test_data_len, 2)
        else:
            accuracy = round(all_true_count * 100 / (test_data_len * sequence_len), 2)

        logger.info("整体准确率: {}%".format(accuracy))

        # 构建总结
        blue_summary = []
        blue_summary.append("蓝球模型评估结果")
        blue_summary.append("=" * 50)
        blue_summary.append(f"测试期数: {test_data_len}")
        blue_summary.append(f"使用指定号码: {enhancer.specified_blue_numbers}")

        blue_summary.append("CRF预测结果:")
        for k, v in sorted(eval_d.items()):
            blue_summary.append("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        blue_summary.append("复式预测结果 (Top{}):".format(blue_topk))
        for k, v in sorted(eval_topk_d.items()):
            blue_summary.append("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))

        blue_summary.append("整体准确率: {}%".format(accuracy))

        return "\n".join(blue_summary)



def run(name, train_test_split=0.65):
    """增强的训练执行函数（修复保存逻辑）"""
    logger.info("正在创建【{}】增强训练集和测试集...".format(name_path[name]["name"]))

    train_data, test_data = create_train_test_data(
        name, model_args[name]["model_args"]["windows_size"], train_test_split
    )

    logger.info("开始增强训练【{}】红球模型...".format(name_path[name]["name"]))
    red_summary = train_with_eval_red_ball_model(
        name,
        x_train=train_data["red"]["x_data"],
        y_train=train_data["red"]["y_data"],
        x_test=test_data["red"]["x_data"],
        y_test=test_data["red"]["y_data"],
    )

    logger.info("开始增强训练【{}】蓝球模型...".format(name_path[name]["name"]))
    blue_summary = train_with_eval_blue_ball_model(
        name,
        x_train=train_data["blue"]["x_data"],
        y_train=train_data["blue"]["y_data"],
        x_test=test_data["blue"]["x_data"],
        y_test=test_data["blue"]["y_data"]
    )

    # 关键修改：确保节点映射文件正确保存
    key_name_path = f"{model_path}/{name}/{pred_key_name}"
    os.makedirs(os.path.dirname(key_name_path), exist_ok=True)

    # 保存节点映射文件
    with open(key_name_path, 'w', encoding='utf-8') as f:
        json.dump(pred_key, f, ensure_ascii=False, indent=2)

    logger.info(f"节点映射文件已保存到: {key_name_path}")
    logger.info(f"节点映射内容: {pred_key}")

    # 将红球和蓝球的总结内容合并并保存到文件
    # 获取本次训练使用的epoch轮数
    red_epochs = model_args[name]["model_args"]["red_epochs"]
    blue_epochs = model_args[name]["model_args"]["blue_epochs"]
    
    summary_content = f"模型训练总结 - {name_path[name]['name']}\n"
    summary_content += "=" * 60 + "\n\n"
    summary_content += f"训练配置: 红球训练epoch - {red_epochs}, 蓝球训练epoch - {blue_epochs}\n"
    summary_content += f"红球指定号码: {SPECIFIED_RED_NUMBERS}\n"
    summary_content += f"蓝球指定号码: {SPECIFIED_BLUE_NUMBERS}\n\n"
    summary_content += "红球模型评估结果:\n"
    summary_content += red_summary + "\n\n" if red_summary else "红球模型评估未完成\n\n"
    summary_content += "蓝球模型评估结果:\n"
    summary_content += blue_summary + "\n\n" if blue_summary else "蓝球模型评估未完成\n\n"

    save_summary_to_file(name, summary_content)

    logger.info("【{}】模型训练和评估完成！".format(name_path[name]["name"]))


# 测试函数
def test_model():
    """测试模型功能"""
    logger.info("开始测试彩票预测模型...")

    try:
        # 测试双色球模型
        run("ssq", 0.7)
        logger.info("测试完成！")
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球")
    parser.add_argument('--train_test_split', default=0.65, type=float, help="训练集占比")
    args = parser.parse_args()

    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        # 检查必要的配置
        if args.name not in name_path:
            raise Exception(f"不支持的玩法名称: {args.name}")

        # 运行训练
        run(args.name, args.train_test_split)