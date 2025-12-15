# -*- coding:utf-8 -*-
"""
Author: BigCat
Enhanced with AC Value, Zone Analysis, and Genetic Algorithm
Fixed Index Error: indices[0,0] = -30 is not in [0, 1089]
Fixed Dimension Mismatch Error in Genetic Algorithm
Fixed Blue Ball Prediction for DLT (2 main predictions + 5 recommended)
修改为使用指定号码集合进行预测
"""
import argparse
import json
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from config import *
from get_data import get_current_number, spider
from loguru import logger
import random
from typing import List, Dict, Tuple
import os
import sys


# 自定义日志配置
def setup_logging(name="lottery"):
    """设置日志配置"""
    logger.remove()

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{current_time}.log")

    # 文件日志 - 所有详细信息
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8"
    )

    # 控制台日志 - 只显示关键信息
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        filter=lambda record: any(
            keyword in record["message"] for keyword in ["红球", "蓝球", "期号", "预测", "命中", "AC值", "断区", "模型", "加载", "完成"])
    )

    return logger


# 初始化日志
logger = setup_logging("dlt_predict")

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
args = parser.parse_args()

# 确保必要的配置存在
if not args.name:
    raise Exception("玩法名称不能为空！")
else:
    if args.name not in name_path:
        raise Exception(f"不支持的玩法名称: {args.name}")


class EnhancedPredictor:
    """增强的预测器，集成三种算法，使用指定号码集合"""
    
    def __init__(self, name: str):
        self.name = name
        self.ac_config = model_args[name]["algorithm_config"]["ac_config"]
        self.zone_config = model_args[name]["algorithm_config"]["zone_config"]
        
        # 使用指定号码
        self.specified_red_numbers = sorted(SPECIFIED_RED_NUMBERS)
        self.specified_blue_numbers = sorted(SPECIFIED_BLUE_NUMBERS)
        
        # 创建映射字典
        self.red_mapping = {num: idx for idx, num in enumerate(self.specified_red_numbers)}
        self.blue_mapping = {num: idx for idx, num in enumerate(self.specified_blue_numbers)}
        
        # 反向映射
        self.red_reverse_mapping = {idx: num for idx, num in enumerate(self.specified_red_numbers)}
        self.blue_reverse_mapping = {idx: num for idx, num in enumerate(self.specified_blue_numbers)}
        
        logger.info(f"预测器使用指定号码 - 红球: {self.specified_red_numbers}")
        logger.info(f"预测器使用指定号码 - 蓝球: {self.specified_blue_numbers}")

    def map_to_specified_index(self, numbers: List[int], ball_type: str) -> List[int]:
        """将实际号码映射到指定号码的索引"""
        if ball_type == "red":
            mapping = self.red_mapping
        else:
            mapping = self.blue_mapping
            
        try:
            return [mapping[num] for num in numbers]
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

    def filter_predictions_by_specified_set(self, candidate_predictions: List[List[int]], 
                                          ball_type: str) -> List[List[int]]:
        """根据指定集合过滤预测结果"""
        if ball_type == "red":
            specified_set = set(self.specified_red_numbers)
        else:
            specified_set = set(self.specified_blue_numbers)
            
        filtered_predictions = []
        for numbers in candidate_predictions:
            if all(num in specified_set for num in numbers):
                filtered_predictions.append(numbers)
                
        return filtered_predictions

    def rank_predictions_by_algorithm(self, candidate_predictions: List[List[int]],
                                    ball_type: str) -> List[Dict]:
        """根据算法指标对预测结果进行排名"""
        # 先过滤掉不在指定集合的预测
        candidate_predictions = self.filter_predictions_by_specified_set(candidate_predictions, ball_type)
        
        if not candidate_predictions:
            logger.warning(f"没有有效的{ball_type}球预测结果")
            return []
        
        ranked_predictions = []
        
        for i, numbers in enumerate(candidate_predictions):
            # 计算AC值
            ac_value = self.calculate_ac_value(numbers)
            
            # 计算AC值得分（越接近最优AC值得分越高）
            optimal_ac = self.ac_config[ball_type]["optimal_ac"]
            ac_score = 1.0 - min(abs(ac_value - optimal_ac) / optimal_ac, 1.0)
            
            # 计算断区得分
            zone_score = self.calculate_zone_score(numbers, ball_type)
            
            # 综合得分
            total_score = ac_score * 0.6 + zone_score * 0.4
            
            ranked_predictions.append({
                'numbers': numbers,
                'ac_value': ac_value,
                'ac_score': ac_score,
                'zone_score': zone_score,
                'total_score': total_score,
                'rank': i + 1
            })
        
        # 按总分排序
        ranked_predictions.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 重新分配排名
        for i, pred in enumerate(ranked_predictions):
            pred['rank'] = i + 1
        
        return ranked_predictions


def get_correct_node_names(pred_key_d, ball_type, name):
    """获取正确的节点名称，修复重复作用域问题"""
    prefix = "red_" if ball_type == "red" else "blue_"
    ball_key = ball_name[0][0] if ball_type == "red" else ball_name[1][0]

    new_key = f"{prefix}{ball_key}"
    new_logits_key = f"{prefix}{ball_key}_logits"

    if new_key in pred_key_d and new_logits_key in pred_key_d:
        node_name = pred_key_d[new_key]
        logits_name = pred_key_d[new_logits_key]

        # 修复重复作用域问题
        if name == "dlt" and ball_type == "blue" and "dlt_blue_model/dlt_blue_model/" in node_name:
            node_name = node_name.replace("dlt_blue_model/dlt_blue_model/", "dlt_blue_model/")
            logits_name = logits_name.replace("dlt_blue_model/dlt_blue_model/", "dlt_blue_model/")
            logger.info(f"修复{ball_type}球节点名称: {node_name}")

        return node_name, logits_name

    # 回退到旧命名规则
    old_key = ball_key
    old_logits_key = f"{ball_key}_logits"

    if old_key in pred_key_d and old_logits_key in pred_key_d:
        return pred_key_d[old_key], pred_key_d[old_logits_key]

    # 默认值
    default_node = "ReverseSequence_1:0"
    default_logits = "dense/BiasAdd:0"
    logger.warning(f"使用默认节点名称: {default_node}")
    return default_node, default_logits


def load_model(name):
    """加载模型 - 修复版本"""
    try:
        import os

        # 修复路径问题
        red_path = model_args[name]["path"]["red"]
        blue_path = model_args[name]["path"]["blue"]

        # 确保路径以斜杠结尾
        if not red_path.endswith('/') and not red_path.endswith('\\'):
            red_path += '/'
        if not blue_path.endswith('/') and not blue_path.endswith('\\'):
            blue_path += '/'

        # 更新配置中的路径
        model_args[name]["path"]["red"] = red_path
        model_args[name]["path"]["blue"] = blue_path

        logger.info(f"红球模型路径: {red_path}")
        logger.info(f"蓝球模型路径: {blue_path}")

        # 检查红球模型文件 - 修复逻辑
        red_ckpt_path = f"{red_path}red_ball_model.ckpt"
        red_meta_path = f"{red_path}red_ball_model.ckpt.meta"

        logger.info(f"检查红球模型文件: {red_ckpt_path}")
        logger.info(f"检查红球元文件: {red_meta_path}")

        # 修复：正确查找模型文件
        if not os.path.exists(red_ckpt_path + ".index"):
            if os.path.exists(red_path):
                files = os.listdir(red_path)
                logger.info(f"红球模型目录内容: {files}")

                # 查找正确的模型文件
                ckpt_files = [f for f in files if f.startswith('red_ball_model') and f.endswith('.index')]
                if ckpt_files:
                    # 使用.index文件来推断模型前缀
                    model_prefix = ckpt_files[0].replace('.index', '')
                    red_ckpt_path = f"{red_path}{model_prefix}"
                    red_meta_path = f"{red_path}{model_prefix}.meta"
                    logger.info(f"使用实际红球模型前缀: {model_prefix}")
                else:
                    # 如果没有找到.index文件，尝试其他模式
                    ckpt_base_files = [f for f in files if f.startswith('red_ball_model') and not f.endswith('.meta')]
                    if ckpt_base_files:
                        # 使用第一个非.meta文件作为基础
                        base_file = ckpt_base_files[0]
                        model_prefix = base_file.split('.')[0]  # 取文件名部分
                        red_ckpt_path = f"{red_path}{model_prefix}"
                        red_meta_path = f"{red_path}{model_prefix}.meta"
                        logger.info(f"使用基础红球模型文件: {model_prefix}")

        # 检查文件是否存在
        if not os.path.exists(red_meta_path):
            raise FileNotFoundError(f"红球元文件不存在: {red_meta_path}")

        # 同样的修复逻辑应用于蓝球模型
        blue_ckpt_path = f"{blue_path}blue_ball_model.ckpt"
        blue_meta_path = f"{blue_path}blue_ball_model.ckpt.meta"

        logger.info(f"检查蓝球模型文件: {blue_ckpt_path}")
        logger.info(f"检查蓝球元文件: {blue_meta_path}")

        if not os.path.exists(blue_ckpt_path + ".index"):
            if os.path.exists(blue_path):
                files = os.listdir(blue_path)
                logger.info(f"蓝球模型目录内容: {files}")

                ckpt_files = [f for f in files if f.startswith('blue_ball_model') and f.endswith('.index')]
                if ckpt_files:
                    model_prefix = ckpt_files[0].replace('.index', '')
                    blue_ckpt_path = f"{blue_path}{model_prefix}"
                    blue_meta_path = f"{blue_path}{model_prefix}.meta"
                    logger.info(f"使用实际蓝球模型前缀: {model_prefix}")
                else:
                    ckpt_base_files = [f for f in files if f.startswith('blue_ball_model') and not f.endswith('.meta')]
                    if ckpt_base_files:
                        base_file = ckpt_base_files[0]
                        model_prefix = base_file.split('.')[0]
                        blue_ckpt_path = f"{blue_path}{model_prefix}"
                        blue_meta_path = f"{blue_path}{model_prefix}.meta"
                        logger.info(f"使用基础蓝球模型文件: {model_prefix}")

        if not os.path.exists(blue_meta_path):
            raise FileNotFoundError(f"蓝球元文件不存在: {blue_meta_path}")

        # 加载红球模型 - 使用正确的路径
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(red_meta_path)
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, red_ckpt_path)
        logger.info("已加载红球模型！")

        # 加载蓝球模型
        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(blue_meta_path)
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, blue_ckpt_path)
        logger.info("已加载蓝球模型！")

        # 加载关键节点名
        key_name_path = f"{model_path}/{name}/{pred_key_name}"
        logger.info(f"加载关键节点文件: {key_name_path}")

        if not os.path.exists(key_name_path):
            possible_paths = [
                f"{red_path}{pred_key_name}",
                f"{blue_path}{pred_key_name}",
                f"{model_path}/{name}/{pred_key_name}",
                f"{model_path}{pred_key_name}"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    key_name_path = path
                    logger.info(f"找到关键节点文件: {path}")
                    break
            else:
                raise FileNotFoundError(f"关键节点文件不存在，无法进行预测。请检查以下路径: {possible_paths}")

        with open(key_name_path, 'r', encoding='utf-8') as f:
            pred_key_d = json.load(f)
            logger.info(f"加载的关键节点: {pred_key_d}")

        # 获取红球和蓝球的正确节点名称
        red_node, red_logits_node = get_correct_node_names(pred_key_d, "red", name)
        blue_node, blue_logits_node = get_correct_node_names(pred_key_d, "blue", name)

        # 更新节点映射字典
        corrected_pred_key_d = {
            "红球": red_node,
            "红球_logits": red_logits_node,
            "蓝球": blue_node,
            "蓝球_logits": blue_logits_node
        }

        logger.info(f"校正后的节点映射: {corrected_pred_key_d}")

        current_number = get_current_number(name)
        logger.info("【{}】最近一期:{}".format(name_path[name]["name"], current_number))
        return red_graph, red_sess, blue_graph, blue_sess, corrected_pred_key_d, current_number

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise Exception(f"模型加载失败: {e}")


def get_year():
    """截取年份"""
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(mode, name, predict_features, windows_size):
    """处理异常"""
    if mode:
        return predict_features
    else:
        if len(predict_features) != windows_size:
            logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
            last_current_year = (get_year() - 1) * 1000
            max_times = 160
            while len(predict_features) != windows_size:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[
                    [x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features


def get_topk_predictions(probabilities, topk, n_class, enhancer=None, ball_type="red"):
    """获取概率最高的topk个预测号码"""
    if probabilities.ndim == 3:
        avg_probs = np.mean(probabilities[0], axis=0)
    elif probabilities.ndim == 2:
        avg_probs = probabilities[0]
    elif probabilities.ndim == 1:
        avg_probs = probabilities
    else:
        avg_probs = np.ones(n_class) / n_class

    if len(avg_probs) < n_class:
        avg_probs = np.pad(avg_probs, (0, n_class - len(avg_probs)), mode='constant', constant_values=0.01)
    elif len(avg_probs) > n_class:
        avg_probs = avg_probs[:n_class]

    if np.sum(avg_probs) > 0:
        avg_probs = avg_probs / np.sum(avg_probs)
    else:
        avg_probs = np.ones(n_class) / n_class

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


def map_input_data_to_indices(data_array, enhancer, ball_type="red"):
    """将输入数据中的实际号码映射到指定号码的索引"""
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


def get_red_ball_predict_result(red_graph, red_sess, pred_key_d, predict_features, sequence_len, windows_size):
    """获取红球预测结果 - 修复节点名称问题，使用指定号码"""
    try:
        name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
        data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int)
        
        logger.info(f"红球输入数据形状: {data.shape}")
        logger.info(f"红球原始数据示例: {data[0]}")

        # 创建增强器用于映射
        enhancer = EnhancedPredictor("dlt")
        
        # 将实际号码映射到指定号码的索引
        data_mapped = map_input_data_to_indices(data.reshape(1, windows_size, sequence_len), enhancer, "red")
        
        logger.info(f"映射后数据形状: {data_mapped.shape}")
        logger.info(f"映射后数据示例: {data_mapped[0]}")

        with red_graph.as_default():
            # 获取预测节点
            pred_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d["红球"])

            # 动态查找输入节点名称
            input_tensor_name = None
            sequence_length_name = None

            # 遍历图中的操作，查找输入节点
            for op in tf.compat.v1.get_default_graph().get_operations():
                if "input" in op.name.lower() and "red" in op.name.lower():
                    if len(op.outputs) > 0:
                        input_tensor_name = op.outputs[0].name
                        logger.info(f"找到输入节点: {input_tensor_name}")

                if "sequence_length" in op.name.lower() and "red" in op.name.lower():
                    if len(op.outputs) > 0:
                        sequence_length_name = op.outputs[0].name
                        logger.info(f"找到序列长度节点: {sequence_length_name}")

            # 如果没找到，尝试可能的输入节点名称
            if not input_tensor_name:
                possible_input_names = [
                    "dlt_red_model/inputs:0",
                    "inputs:0",
                    "input:0",
                    "dlt_red_model/input:0",
                    "dlt_red_model/dlt_red_model/inputs:0"
                ]
                for name in possible_input_names:
                    try:
                        tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                        input_tensor_name = name
                        logger.info(f"使用输入节点: {input_tensor_name}")
                        break
                    except:
                        continue

            if not sequence_length_name:
                possible_seq_names = [
                    "dlt_red_model/sequence_length:0",
                    "sequence_length:0",
                    "dlt_red_model/dlt_red_model/sequence_length:0"
                ]
                for name in possible_seq_names:
                    try:
                        tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                        sequence_length_name = name
                        logger.info(f"使用序列长度节点: {sequence_length_name}")
                        break
                    except:
                        continue

            if not input_tensor_name:
                raise Exception("无法找到输入节点")

            # 准备feed_dict
            feed_dict = {}
            feed_dict[input_tensor_name] = data_mapped

            if sequence_length_name:
                feed_dict[sequence_length_name] = np.array([sequence_len] * 1)

            # 尝试获取logits节点
            try:
                logits_tensor_name = pred_key_d.get("红球_logits")
                if logits_tensor_name:
                    logits_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(logits_tensor_name)
                    probabilities = tf.nn.softmax(logits_tensor)

                    result = red_sess.run({
                        'prediction': pred_tensor,
                        'probabilities': probabilities
                    }, feed_dict=feed_dict)
                else:
                    raise Exception("没有logits节点名称")

            except Exception as e:
                logger.warning(f"无法获取logits节点: {e}")
                # 直接使用预测结果
                pred_result = red_sess.run(pred_tensor, feed_dict=feed_dict)

                # 手动创建概率分布
                n_class = 25  # 红球25个指定号码
                probs = np.ones(n_class) / n_class

                result = {
                    'prediction': pred_result,
                    'probabilities': probs.reshape(1, 1, n_class)
                }

        pred = result['prediction']
        probs = result['probabilities']

        logger.info(f"红球预测结果形状: {pred.shape}")
        logger.info(f"红球概率结果形状: {probs.shape}")

        # 处理概率数组
        if not hasattr(probs, 'shape') or probs.ndim == 0:
            probs = np.array([probs])

        if probs.ndim == 4:
            avg_probs = np.mean(probs[0], axis=(0, 1))
        elif probs.ndim == 3:
            avg_probs = np.mean(probs[0], axis=0)
        elif probs.ndim == 2:
            avg_probs = probs[0]
        elif probs.ndim == 1:
            avg_probs = probs
        else:
            avg_probs = np.ones(25) / 25

        # 确保概率数组长度正确
        n_class = 25  # 红球25个指定号码
        if len(avg_probs) < n_class:
            avg_probs = np.pad(avg_probs, (0, n_class - len(avg_probs)), mode='constant', constant_values=0.01)
        elif len(avg_probs) > n_class:
            avg_probs = avg_probs[:n_class]

        # 归一化概率
        if np.sum(avg_probs) > 0:
            avg_probs = avg_probs / np.sum(avg_probs)
        else:
            avg_probs = np.ones(n_class) / n_class

        # 获取概率最高的13个号码（映射回指定号码）
        top_indices = np.argsort(avg_probs)[-13:][::-1]
        top_numbers = [enhancer.red_reverse_mapping[idx] for idx in top_indices]
        top_probs = [avg_probs[idx] for idx in top_indices]

        logger.info(f"红球Top13号码: {top_numbers}")
        logger.info(f"红球Top13概率: {top_probs}")

        # 将预测结果也映射回指定号码
        pred_numbers = enhancer.map_from_specified_index(pred[0].tolist(), "red")
        logger.info(f"红球CRF预测号码: {pred_numbers}")

        return top_numbers, top_probs, pred_numbers

    except Exception as e:
        logger.error(f"红球预测错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise Exception(f"红球预测失败: {e}")


def get_blue_ball_predict_result(blue_graph, blue_sess, pred_key_d, name, predict_features, sequence_len, windows_size):
    """获取蓝球预测结果 - 修复大乐透蓝球维度问题和节点名称问题，使用指定号码"""
    try:
        enhancer = EnhancedPredictor(name)
        
        if name == "ssq":
            # 双色球蓝球处理
            data = predict_features[[ball_name[1][0]]].values.astype(int)
            logger.info(f"双色球蓝球输入数据形状: {data.shape}")
            logger.info(f"双色球蓝球原始数据示例: {data[0]}")

            # 将实际号码映射到指定号码的索引
            data_mapped = map_input_data_to_indices(data, enhancer, "blue")
            
            logger.info(f"映射后数据形状: {data_mapped.shape}")

            with blue_graph.as_default():
                # 使用校正后的节点名称
                pred_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d["蓝球"])

                # 动态查找输入节点名称
                input_tensor_name = None

                # 遍历图中的操作，查找输入节点
                for op in tf.compat.v1.get_default_graph().get_operations():
                    if "input" in op.name.lower() and "blue" in op.name.lower():
                        if len(op.outputs) > 0:
                            input_tensor_name = op.outputs[0].name
                            logger.info(f"找到蓝球输入节点: {input_tensor_name}")
                            break

                # 如果没找到，尝试可能的输入节点名称
                if not input_tensor_name:
                    possible_input_names = [
                        "dlt_blue_model/inputs:0",
                        "inputs:0",
                        "input:0",
                        "dlt_blue_model/input:0",
                        "dlt_blue_model/dlt_blue_model/inputs:0"
                    ]
                    for name in possible_input_names:
                        try:
                            tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                            input_tensor_name = name
                            logger.info(f"使用蓝球输入节点: {input_tensor_name}")
                            break
                        except:
                            continue

                if not input_tensor_name:
                    raise Exception("无法找到蓝球输入节点")

                # 尝试获取logits节点
                try:
                    logits_tensor_name = pred_key_d.get("蓝球_logits")
                    if logits_tensor_name:
                        logits_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(logits_tensor_name)
                        probabilities = tf.nn.softmax(logits_tensor)

                        result = blue_sess.run({
                            'prediction': pred_tensor,
                            'probabilities': probabilities
                        }, feed_dict={
                            input_tensor_name: data_mapped.reshape(1, windows_size)
                        })
                    else:
                        raise Exception("没有logits节点名称")

                except Exception as e:
                    logger.warning(f"无法获取logits节点: {e}")
                    pred_result = blue_sess.run(pred_tensor, feed_dict={
                        input_tensor_name: data_mapped.reshape(1, windows_size)
                    })

                    # 手动创建概率分布
                    n_class = 10  # 蓝球10个指定号码
                    probs = np.zeros(n_class)

                    pred = pred_result[0]
                    if pred < n_class:
                        probs[pred] = 1.0
                    else:
                        probs = np.ones(n_class) / n_class

                    result = {
                        'prediction': pred_result,
                        'probabilities': probs.reshape(1, n_class)
                    }

            probs = result['probabilities'][0]  # 去除batch维度
            pred = result['prediction'][0]

            # 双色球：返回1个CRF预测和5个复式推荐
            crf_number = enhancer.map_from_specified_index([pred], "blue")[0]
            top_indices = np.argsort(probs)[-5:][::-1]
            top_numbers = [enhancer.blue_reverse_mapping[idx] for idx in top_indices]
            top_probs = [probs[idx] for idx in top_indices]

            logger.info(f"双色球蓝球CRF预测号码: {crf_number}")
            logger.info(f"双色球蓝球复式推荐Top5: {top_numbers}")

            return [crf_number], top_numbers, top_probs

        else:
            # 大乐透蓝球处理 - 修复维度问题
            name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
            data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int)

            logger.info(f"大乐透蓝球输入数据形状: {data.shape}")
            logger.info(f"大乐透蓝球原始数据示例: {data[0]}")

            # 将实际号码映射到指定号码的索引
            data_mapped = map_input_data_to_indices(data.reshape(1, windows_size, sequence_len), enhancer, "blue")

            logger.info(f"修复后的大乐透蓝球输入数据形状: {data_mapped.shape}")

            with blue_graph.as_default():
                # 获取预测节点
                pred_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d["蓝球"])

                # 动态查找输入节点名称
                input_tensor_name = None
                sequence_length_name = None

                # 遍历图中的操作，查找输入节点
                for op in tf.compat.v1.get_default_graph().get_operations():
                    if "input" in op.name.lower() and "blue" in op.name.lower():
                        if len(op.outputs) > 0:
                            input_tensor_name = op.outputs[0].name
                            logger.info(f"找到蓝球输入节点: {input_tensor_name}")

                    if "sequence_length" in op.name.lower() and "blue" in op.name.lower():
                        if len(op.outputs) > 0:
                            sequence_length_name = op.outputs[0].name
                            logger.info(f"找到蓝球序列长度节点: {sequence_length_name}")

                # 如果没找到，尝试可能的输入节点名称
                if not input_tensor_name:
                    possible_input_names = [
                        "dlt_blue_model/inputs:0",
                        "inputs:0",
                        "input:0",
                        "dlt_blue_model/input:0",
                        "dlt_blue_model/dlt_blue_model/inputs:0"
                    ]
                    for name in possible_input_names:
                        try:
                            tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                            input_tensor_name = name
                            logger.info(f"使用蓝球输入节点: {input_tensor_name}")
                            break
                        except:
                            continue

                if not sequence_length_name:
                    possible_seq_names = [
                        "dlt_blue_model/sequence_length:0",
                        "sequence_length:0",
                        "dlt_blue_model/dlt_blue_model/sequence_length:0"
                    ]
                    for name in possible_seq_names:
                        try:
                            tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(name)
                            sequence_length_name = name
                            logger.info(f"使用蓝球序列长度节点: {sequence_length_name}")
                            break
                        except:
                            continue

                if not input_tensor_name:
                    raise Exception("无法找到蓝球输入节点")

                # 关键修复：动态确定sequence_length
                actual_sequence_len = data_mapped.shape[2]  # 实际的特征数

                # 准备feed_dict
                feed_dict = {}
                feed_dict[input_tensor_name] = data_mapped

                if sequence_length_name:
                    feed_dict[sequence_length_name] = np.array([actual_sequence_len] * 1)

                # 尝试获取logits节点
                try:
                    logits_tensor_name = pred_key_d.get("蓝球_logits")
                    if logits_tensor_name:
                        logits_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(logits_tensor_name)
                        probabilities = tf.nn.softmax(logits_tensor)

                        result = blue_sess.run({
                            'prediction': pred_tensor,
                            'probabilities': probabilities
                        }, feed_dict=feed_dict)
                    else:
                        raise Exception("没有logits节点名称")

                except Exception as e:
                    logger.warning(f"使用标准方法失败: {e}，尝试备用方法")

                    # 备用方法：直接使用预测结果
                    pred_result = blue_sess.run(pred_tensor, feed_dict=feed_dict)

                    # 手动创建概率分布
                    n_class = 10  # 大乐透蓝球共10个指定号码
                    probs = np.ones(n_class) / n_class  # 均匀分布

                    # 根据预测结果调整概率
                    pred = pred_result[0]
                    for i in range(pred.shape[0]):
                        if pred[i] < n_class:
                            probs[pred[i]] += 0.5  # 增加预测号码的概率

                    # 归一化
                    probs = probs / np.sum(probs)

                    result = {
                        'prediction': pred_result,
                        'probabilities': probs.reshape(1, 1, n_class)
                    }

            probs = result['probabilities']
            pred = result['prediction']

            logger.info(f"大乐透蓝球预测结果: {pred}")
            logger.info(f"大乐透蓝球概率结果形状: {probs.shape}")

            # 处理概率数组，得到10个号码的概率分布
            if probs.ndim == 3:
                # 假设形状为 (batch, sequence, class)
                if probs.shape[0] == 1:
                    batch_probs = probs[0]  # 去除batch维度

                    if batch_probs.shape[0] == 2 and batch_probs.shape[1] == 10:
                        # 两个位置的概率分布，求平均
                        avg_probs = np.mean(batch_probs, axis=0)
                    elif batch_probs.shape[0] == 10 and batch_probs.shape[1] == 2:
                        # 转置后求平均
                        avg_probs = np.mean(batch_probs.T, axis=0)
                    else:
                        # 其他形状，直接展平
                        avg_probs = batch_probs.flatten()[:10]  # 只取前10个
                else:
                    avg_probs = np.mean(probs, axis=(0, 1))[:10]  # 对batch和sequence求平均
            elif probs.ndim == 2:
                if probs.shape[0] == 2 and probs.shape[1] == 10:
                    avg_probs = np.mean(probs, axis=0)
                elif probs.shape[0] == 10 and probs.shape[1] == 2:
                    avg_probs = np.mean(probs, axis=1)
                else:
                    avg_probs = probs.flatten()[:10]
            else:
                # 1维或其它，假设是10个元素
                if probs.size >= 10:
                    avg_probs = probs.reshape(-1)[:10]
                else:
                    avg_probs = np.ones(10) / 10

            # 确保avg_probs长度为10
            if len(avg_probs) != 10:
                avg_probs = np.ones(10) / 10

            # 归一化
            if np.sum(avg_probs) > 0:
                avg_probs = avg_probs / np.sum(avg_probs)
            else:
                avg_probs = np.ones(10) / 10

            # 获取CRF预测的2个主要号码（映射回指定号码）
            crf_numbers = []
            for num in pred[0]:
                if num < 10:  # 确保在0-9范围内
                    crf_numbers.append(enhancer.blue_reverse_mapping[num])
                else:
                    # 如果超出范围，使用第一个指定号码
                    crf_numbers.append(enhancer.specified_blue_numbers[0])

            # 去除重复号码
            crf_numbers = list(dict.fromkeys(crf_numbers))[:2]

            # 获取概率最高的5个复式推荐号码
            top_indices = np.argsort(avg_probs)[-5:][::-1]
            top_numbers = [enhancer.blue_reverse_mapping[idx] for idx in top_indices]
            top_probs = [avg_probs[idx] for idx in top_indices]

            logger.info(f"大乐透蓝球CRF预测号码: {crf_numbers}")
            logger.info(f"大乐透蓝球复式推荐Top5: {top_numbers}")
            logger.info(f"大乐透蓝球Top5概率: {[f'{p:.3f}' for p in top_probs]}")

            # 返回CRF预测的2个号码和复式推荐的5个号码
            return crf_numbers, top_numbers, top_probs

    except Exception as e:
        logger.error(f"蓝球预测错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise Exception(f"蓝球预测失败: {e}")


def get_enhanced_final_result(red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features, mode=0):
    """增强的最终预测函数 - 修复版本，使用指定号码"""
    try:
        m_args = model_args[name]["model_args"]
        
        # 创建增强预测器
        enhancer = EnhancedPredictor(name)

        # 获取基础预测结果
        if name == "ssq":
            # 双色球处理
            red_top_numbers, red_top_probs, red_crf_numbers = get_red_ball_predict_result(
                red_graph, red_sess, pred_key_d,
                predict_features, m_args["sequence_len"], m_args["windows_size"]
            )
            blue_crf_numbers, blue_top_numbers, blue_top_probs = get_blue_ball_predict_result(
                blue_graph, blue_sess, pred_key_d,
                name, predict_features, 1, m_args["windows_size"]  # 双色球蓝球序列长度为1
            )
        else:
            # 大乐透处理 - 修复序列长度
            red_top_numbers, red_top_probs, red_crf_numbers = get_red_ball_predict_result(
                red_graph, red_sess, pred_key_d,
                predict_features, m_args["red_sequence_len"], m_args["windows_size"]
            )
            blue_crf_numbers, blue_top_numbers, blue_top_probs = get_blue_ball_predict_result(
                blue_graph, blue_sess, pred_key_d,
                name, predict_features, m_args["blue_sequence_len"], m_args["windows_size"]
            )

        # 对红球预测结果进行算法排名
        red_candidates = []
        # 生成正确数量的红球组合
        target_red_count = 5 if name == "dlt" else 6

        # 从top13中生成多个候选组合
        for i in range(min(10, len(red_top_numbers) - target_red_count + 1)):
            candidate = red_top_numbers[i:i + target_red_count]
            if len(candidate) == target_red_count:
                red_candidates.append(candidate)

        # 也添加CRF预测的组合
        if len(red_crf_numbers) == target_red_count:
            red_candidates.append(red_crf_numbers)

        ranked_red_predictions = enhancer.rank_predictions_by_algorithm(red_candidates, "red")

        # 对蓝球预测结果进行算法排名
        blue_candidates = []
        # 生成正确数量的蓝球组合
        target_blue_count = 2 if name == "dlt" else 1

        if target_blue_count == 1:
            # 单蓝球：每个号码单独作为候选
            for num in blue_top_numbers:
                blue_candidates.append([num])
            # 也添加CRF预测
            for num in blue_crf_numbers:
                blue_candidates.append([num])
        else:
            # 双蓝球：生成所有可能的2个号码组合
            for i in range(len(blue_top_numbers)):
                for j in range(i + 1, len(blue_top_numbers)):
                    blue_candidates.append([blue_top_numbers[i], blue_top_numbers[j]])
            # 也添加CRF预测
            if len(blue_crf_numbers) == 2:
                blue_candidates.append(blue_crf_numbers)

        ranked_blue_predictions = enhancer.rank_predictions_by_algorithm(blue_candidates, "blue")

        return {
            "red_ball": {
                "base_prediction": red_top_numbers,
                "base_probabilities": red_top_probs,
                "crf_prediction": red_crf_numbers,
                "enhanced_predictions": ranked_red_predictions,
                "best_enhanced": ranked_red_predictions[0] if ranked_red_predictions else None
            },
            "blue_ball": {
                "crf_prediction": blue_crf_numbers,  # CRF直接预测的号码
                "top_prediction": blue_top_numbers,  # 复式推荐的Top5号码
                "top_probabilities": blue_top_probs,  # Top5号码的概率
                "enhanced_predictions": ranked_blue_predictions,
                "best_enhanced": ranked_blue_predictions[0] if ranked_blue_predictions else None
            },
            "specified_numbers": {
                "red": enhancer.specified_red_numbers,
                "blue": enhancer.specified_blue_numbers
            }
        }

    except Exception as e:
        logger.error(f"增强结果处理错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise e


def save_enhanced_prediction_result(name, current_number, result):
    """保存增强预测结果到txt文件 - 修复版本"""
    try:
        filename = f"{name}_enhanced.txt"
        next_period = int(current_number) + 1
        # 获取训练epoch轮数
        red_epochs = model_args[name]["model_args"]["red_epochs"]
        blue_epochs = model_args[name]["model_args"]["blue_epochs"]

        content = f"期号: {next_period} | 训练红球epoch: {red_epochs} | 训练蓝球epoch: {blue_epochs}\n"
        content += "=" * 80 + "\n"
        content += f"{name_path[name]['name']} - 增强预测结果\n"
        content += "=" * 80 + "\n\n"
        
        # 显示指定号码
        content += f"指定红球号码 ({len(result['specified_numbers']['red'])}个): {sorted(result['specified_numbers']['red'])}\n"
        content += f"指定蓝球号码 ({len(result['specified_numbers']['blue'])}个): {sorted(result['specified_numbers']['blue'])}\n"
        content += "-" * 40 + "\n\n"

        # 红球结果
        content += "红球预测分析:\n"
        content += "-" * 40 + "\n"
        content += f"基础预测 (Top13): {result['red_ball']['base_prediction']}\n"
        content += f"CRF直接预测: {sorted(result['red_ball']['crf_prediction'])}\n"

        if result['red_ball']['best_enhanced']:
            best_red = result['red_ball']['best_enhanced']
            content += f"增强预测 (排名第{best_red['rank']}): {sorted(best_red['numbers'])}\n"
            content += f"AC值: {best_red['ac_value']} | 断区得分: {best_red['zone_score']:.3f}\n"
            content += f"综合得分: {best_red['total_score']:.3f}\n"

        content += "\n红球候选预测排名 (前5个):\n"
        for pred in result['red_ball']['enhanced_predictions'][:5]:
            content += f"  排名{pred['rank']}: {sorted(pred['numbers'])} (AC:{pred['ac_value']}, 断区:{pred['zone_score']:.3f}, 总分:{pred['total_score']:.3f})\n"

        content += "\n"

        # 蓝球结果
        content += "蓝球预测分析:\n"
        content += "-" * 40 + "\n"
        content += f"CRF直接预测: {sorted(result['blue_ball']['crf_prediction'])}\n"
        content += f"复式推荐Top5: {sorted(result['blue_ball']['top_prediction'])}\n"
        content += f"Top5概率: {[f'{p:.3f}' for p in result['blue_ball']['top_probabilities']]}\n"

        if result['blue_ball']['best_enhanced']:
            best_blue = result['blue_ball']['best_enhanced']
            content += f"增强预测 (排名第{best_blue['rank']}): {sorted(best_blue['numbers'])}\n"
            if len(best_blue['numbers']) > 1:  # 多蓝球情况
                content += f"AC值: {best_blue['ac_value']} | 断区得分: {best_blue['zone_score']:.3f}\n"
            content += f"综合得分: {best_blue['total_score']:.3f}\n"

        content += "\n蓝球候选预测排名 (前3个):\n"
        for pred in result['blue_ball']['enhanced_predictions'][:3]:
            if len(pred['numbers']) > 1:
                content += f"  排名{pred['rank']}: {sorted(pred['numbers'])} (AC:{pred['ac_value']}, 断区:{pred['zone_score']:.3f}, 总分:{pred['total_score']:.3f})\n"
            else:
                content += f"  排名{pred['rank']}: {pred['numbers']} (断区:{pred['zone_score']:.3f}, 总分:{pred['total_score']:.3f})\n"

        content += "\n算法说明:\n"
        content += "-" * 40 + "\n"
        content += "? CRF预测: 模型直接输出的预测号码\n"
        content += "? 复式推荐: 基于概率分布的前5个最可能号码\n"
        content += "? AC值算法: 优化号码组合的复杂度\n"
        content += "? 断区杀号: 分析各区间的出现概率\n"
        content += "? 综合得分 = AC值得分 × 0.6 + 断区得分 × 0.4\n"
        content += "? 排名越靠前，预测结果越可靠\n"
        content += f"? 所有预测均基于指定号码集合\n\n"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"增强预测结果已保存至 {filename}")
    except Exception as e:
        logger.error(f"保存结果错误: {e}")


def run(name):
    """执行增强预测"""
    try:
        logger.info(f"开始执行【{name_path[name]['name']}】预测")
        logger.info(f"使用指定号码 - 红球: {SPECIFIED_RED_NUMBERS}")
        logger.info(f"使用指定号码 - 蓝球: {SPECIFIED_BLUE_NUMBERS}")

        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number = load_model(name)
        windows_size = model_args[name]["model_args"]["windows_size"]

        data = spider(name, 1, current_number, "predict")
        next_period = int(current_number) + 1

        logger.info(f"预测期号：{next_period}")
        predict_features_ = try_error(1, name, data.iloc[:windows_size], windows_size)

        # 获取预测结果
        pred_result = get_enhanced_final_result(
            red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features_
        )

        # 显示简洁结果
        best_red = pred_result['red_ball']['best_enhanced']
        best_blue = pred_result['blue_ball']['best_enhanced']

        logger.info("=" * 50)
        logger.info("最终预测结果:")
        logger.info(f"指定红球: {sorted(SPECIFIED_RED_NUMBERS)}")
        logger.info(f"指定蓝球: {sorted(SPECIFIED_BLUE_NUMBERS)}")
        if best_red:
            logger.info(
                f"红球推荐: {sorted(best_red['numbers'])} (排名第{best_red['rank']}, 得分:{best_red['total_score']:.3f})")
        if best_blue:
            if len(best_blue['numbers']) > 1:
                logger.info(
                    f"蓝球推荐: {sorted(best_blue['numbers'])} (排名第{best_blue['rank']}, 得分:{best_blue['total_score']:.3f})")
            else:
                logger.info(
                    f"蓝球推荐: {best_blue['numbers'][0]} (排名第{best_blue['rank']}, 得分:{best_blue['total_score']:.3f})")
        logger.info("=" * 50)

        # 保存结果
        save_enhanced_prediction_result(name, current_number, pred_result)
        logger.info("预测完成！")

    except Exception as e:
        logger.error(f"预测过程失败: {e}")


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name)