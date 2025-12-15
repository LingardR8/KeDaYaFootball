# -*- coding:utf-8 -*-
"""
Author: BigCat
Enhanced with AC Value, Zone Analysis, and Genetic Algorithm
修改为使用指定号码集合
"""
import tensorflow as tf
import numpy as np
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood
from config import model_args, SPECIFIED_RED_NUMBERS, SPECIFIED_BLUE_NUMBERS

# 关闭eager模式
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class ACZoneRegularizer:
    """AC值和断区正则化器，使用指定号码集合"""
    
    def __init__(self, name, ball_type):
        self.name = name
        self.ball_type = ball_type
        self.ac_config = model_args[name]["algorithm_config"]["ac_config"][ball_type]
        self.zone_config = model_args[name]["algorithm_config"]["zone_config"][f"{ball_type}_zones"]
        
        # 使用指定号码
        if ball_type == "red":
            self.specified_numbers = sorted(SPECIFIED_RED_NUMBERS)
        else:
            self.specified_numbers = sorted(SPECIFIED_BLUE_NUMBERS)
            
        # 创建映射
        self.mapping = {num: idx for idx, num in enumerate(self.specified_numbers)}
        self.reverse_mapping = {idx: num for idx, num in enumerate(self.specified_numbers)}

    def calculate_ac_value(self, numbers):
        """计算AC值"""
        if len(numbers) < 2:
            return 0

        diffs = set()
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                diffs.add(abs(numbers[i] - numbers[j]))

        return len(diffs) - (len(numbers) - 1)

    def map_to_specified_index(self, numbers):
        """将实际号码映射到指定号码的索引"""
        try:
            return [self.mapping[num] for num in numbers]
        except KeyError:
            # 如果不在指定集合中，返回空列表
            return []

    def map_from_specified_index(self, indices):
        """从指定号码索引映射回实际号码"""
        return [self.reverse_mapping[idx] for idx in indices]

    def ac_regularization(self, y_pred):
        """AC值正则化项"""
        try:
            # 获取预测的号码索引
            pred_indices = tf.argmax(y_pred, axis=-1)
            
            # 将索引映射回实际号码
            def map_indices_to_numbers(indices):
                indices_flat = tf.reshape(indices, [-1])
                numbers = tf.py_function(
                    func=lambda x: [self.reverse_mapping[idx.numpy()] for idx in x],
                    inp=[indices_flat],
                    Tout=tf.int32
                )
                return numbers
            
            def compute_ac_for_sample(indices):
                numbers = map_indices_to_numbers(indices)
                ac_value = tf.py_function(
                    func=lambda x: self.calculate_ac_value(x.numpy()),
                    inp=[numbers],
                    Tout=tf.float32
                )
                return ac_value

            batch_ac_values = tf.map_fn(
                compute_ac_for_sample,
                pred_indices,
                fn_output_signature=tf.float32
            )

            # 计算AC值惩罚
            min_ac = tf.constant(self.ac_config["min_ac"], dtype=tf.float32)
            max_ac = tf.constant(self.ac_config["max_ac"], dtype=tf.float32)

            # 低于最小AC值或高于最大AC值的惩罚
            ac_penalty = tf.reduce_mean(
                tf.maximum(min_ac - batch_ac_values, 0) +  # 低于最小值
                tf.maximum(batch_ac_values - max_ac, 0)  # 高于最大值
            )

            return ac_penalty

        except Exception as e:
            tf.print("AC值正则化计算失败:", e)
            return tf.constant(0.0)

    def zone_regularization(self, y_pred):
        """断区正则化项"""
        try:
            pred_indices = tf.argmax(y_pred, axis=-1)

            def compute_zone_penalty(indices):
                # 将索引映射回实际号码
                indices_flat = tf.reshape(indices, [-1])
                numbers = tf.py_function(
                    func=lambda x: [self.reverse_mapping[idx.numpy()] for idx in x],
                    inp=[indices_flat],
                    Tout=tf.int32
                )
                
                zone_penalty = tf.py_function(
                    func=lambda x: self._calculate_zone_penalty_tf(x.numpy()),
                    inp=[numbers],
                    Tout=tf.float32
                )
                return zone_penalty

            batch_zone_penalties = tf.map_fn(
                compute_zone_penalty,
                pred_indices,
                fn_output_signature=tf.float32
            )

            return tf.reduce_mean(batch_zone_penalties)

        except Exception as e:
            tf.print("断区正则化计算失败:", e)
            return tf.constant(0.0)

    def _calculate_zone_penalty_tf(self, numbers):
        """计算断区惩罚"""
        zone_counts = {zone: 0 for zone in self.zone_config.keys()}

        for num in numbers:
            for zone_name, zone_nums in self.zone_config.items():
                if num in zone_nums:
                    zone_counts[zone_name] += 1
                    break

        # 计算区间覆盖率
        total_zones = len(self.zone_config)
        covered_zones = sum(1 for count in zone_counts.values() if count > 0)

        # 覆盖率越低，惩罚越高
        coverage_ratio = covered_zones / total_zones
        penalty = 1.0 - coverage_ratio

        return np.float32(penalty)

    def specified_numbers_regularization(self, y_pred):
        """指定号码正则化项 - 惩罚非指定号码的预测"""
        try:
            # 获取预测的号码索引
            pred_indices = tf.argmax(y_pred, axis=-1)
            
            # 检查索引是否在有效范围内
            max_valid_idx = len(self.specified_numbers) - 1
            
            def check_valid_indices(indices):
                indices_flat = tf.reshape(indices, [-1])
                # 统计超出范围的索引数量
                out_of_range = tf.reduce_sum(
                    tf.cast(indices_flat > max_valid_idx, tf.float32)
                ) + tf.reduce_sum(
                    tf.cast(indices_flat < 0, tf.float32)
                )
                return out_of_range / tf.cast(tf.size(indices_flat), tf.float32)
            
            batch_penalties = tf.map_fn(
                check_valid_indices,
                pred_indices,
                fn_output_signature=tf.float32
            )
            
            return tf.reduce_mean(batch_penalties)
            
        except Exception as e:
            tf.print("指定号码正则化计算失败:", e)
            return tf.constant(0.0)


class EnhancedLstmWithCRFModel:
    """增强的LSTM+CRF模型，集成算法正则化，使用指定号码集合"""
    
    def __init__(self, batch_size, n_class, ball_num, w_size, embedding_size,
                 words_size, hidden_size, layer_size, name, ball_type):
        # 关键修改：添加变量作用域，确保红球和蓝球使用不同的节点名称
        scope_name = f"{name}_{ball_type}_model"  # 例如: "dlt_red_model" 或 "ssq_blue_model"
        
        with tf.compat.v1.variable_scope(scope_name):
            self._inputs = tf.keras.layers.Input(
                shape=(w_size, ball_num), batch_size=batch_size, name="inputs"
            )
            self._tag_indices = tf.keras.layers.Input(
                shape=(ball_num,), batch_size=batch_size, dtype=tf.int32, name="tag_indices"
            )
            self._sequence_length = tf.keras.layers.Input(
                shape=(), batch_size=batch_size, dtype=tf.int32, name="sequence_length"
            )

            # 构建特征抽取
            embedding = tf.keras.layers.Embedding(words_size, embedding_size)(self._inputs)

            # 对每个球的位置分别处理
            first_lstm = tf.convert_to_tensor(
                [tf.keras.layers.LSTM(hidden_size)(embedding[:, :, i, :]) for i in range(ball_num)]
            )
            first_lstm = tf.transpose(first_lstm, perm=[1, 0, 2])

            # 多层LSTM
            second_lstm = None
            for _ in range(layer_size):
                second_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(first_lstm)

            self._outputs = tf.keras.layers.Dense(n_class)(second_lstm)

            # 构建损失函数
            self._log_likelihood, self._transition_params = crf_log_likelihood(
                self._outputs, self._tag_indices, self._sequence_length
            )
            self._base_loss = tf.reduce_sum(-self._log_likelihood)

            # 构建预测
            self._pred_sequence, self._viterbi_score = crf_decode(
                self._outputs, self._transition_params, self._sequence_length
            )

            # 添加概率输出
            self._probabilities = tf.nn.softmax(self._outputs)

            # 添加算法正则化
            self.regularizer = ACZoneRegularizer(name, ball_type)
            self._enhanced_loss = self._create_enhanced_loss()

    def _create_enhanced_loss(self):
        """创建增强的损失函数"""
        # 基础CRF损失
        base_loss = self._base_loss

        # 算法正则化项
        ac_reg = self.regularizer.ac_regularization(self._outputs)
        zone_reg = self.regularizer.zone_regularization(self._outputs)
        specified_reg = self.regularizer.specified_numbers_regularization(self._outputs)

        # 获取正则化权重
        ac_weight = model_args[self.regularizer.name]["model_args"]["ac_weight"]
        zone_weight = model_args[self.regularizer.name]["model_args"]["zone_weight"]

        # 组合损失 - 添加指定号码正则化
        enhanced_loss = base_loss + ac_weight * ac_reg + zone_weight * zone_reg + 0.5 * specified_reg

        return enhanced_loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def outputs(self):
        return self._outputs

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def base_loss(self):
        return self._base_loss

    @property
    def enhanced_loss(self):
        return self._enhanced_loss

    @property
    def pred_sequence(self):
        return self._pred_sequence

    @property
    def probabilities(self):
        return self._probabilities


class EnhancedSignalLstmModel:
    """增强的单向LSTM模型，集成算法正则化，使用指定号码集合"""
    
    def __init__(self, batch_size, n_class, w_size, embedding_size,
                 hidden_size, outputs_size, layer_size, name, ball_type):
        # 关键修改：添加变量作用域，确保红球和蓝球使用不同的节点名称
        scope_name = f"{name}_{ball_type}_model"  # 例如: "dlt_blue_model" 或 "ssq_blue_model"
        
        with tf.compat.v1.variable_scope(scope_name):
            self._inputs = tf.keras.layers.Input(
                shape=(w_size,), batch_size=batch_size, dtype=tf.int32, name="inputs"
            )
            self._tag_indices = tf.keras.layers.Input(
                shape=(n_class,), batch_size=batch_size, dtype=tf.float32, name="tag_indices"
            )

            embedding = tf.keras.layers.Embedding(outputs_size, embedding_size)(self._inputs)
            lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(embedding)

            # 多层LSTM
            for _ in range(layer_size):
                lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(lstm)

            final_lstm = tf.keras.layers.LSTM(hidden_size, recurrent_dropout=0.2)(lstm)
            self._outputs = tf.keras.layers.Dense(outputs_size, activation="softmax")(final_lstm)

            # 构建损失函数
            self._base_loss = - tf.reduce_sum(self._tag_indices * tf.math.log(self._outputs))

            # 预测结果
            self._pred_label = tf.argmax(self.outputs, axis=1)

            # 添加概率输出
            self._probabilities = self._outputs

            # 添加算法正则化
            self.regularizer = ACZoneRegularizer(name, ball_type)
            self._enhanced_loss = self._create_enhanced_loss()

    def _create_enhanced_loss(self):
        """创建增强的损失函数"""
        # 基础交叉熵损失
        base_loss = self._base_loss

        # 算法正则化项
        ac_reg = self.regularizer.ac_regularization(self._outputs)
        zone_reg = self.regularizer.zone_regularization(self._outputs)
        specified_reg = self.regularizer.specified_numbers_regularization(self._outputs)

        # 获取正则化权重
        ac_weight = model_args[self.regularizer.name]["model_args"]["ac_weight"]
        zone_weight = model_args[self.regularizer.name]["model_args"]["zone_weight"]

        # 组合损失 - 添加指定号码正则化
        enhanced_loss = base_loss + ac_weight * ac_reg + zone_weight * zone_reg + 0.5 * specified_reg

        return enhanced_loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def outputs(self):
        return self._outputs

    @property
    def base_loss(self):
        return self._base_loss

    @property
    def enhanced_loss(self):
        return self._enhanced_loss

    @property
    def pred_label(self):
        return self._pred_label

    @property
    def probabilities(self):
        return self._probabilities


# 保持原有类定义以保持兼容性
class LstmWithCRFModel:
    """lstm + crf解码模型"""
    
    def __init__(self, batch_size, n_class, ball_num, w_size, embedding_size, words_size, hidden_size, layer_size):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size, ball_num), batch_size=batch_size, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(ball_num,), batch_size=batch_size, dtype=tf.int32, name="tag_indices"
        )
        self._sequence_length = tf.keras.layers.Input(
            shape=(), batch_size=batch_size, dtype=tf.int32, name="sequence_length"
        )

        # 构建特征抽取
        embedding = tf.keras.layers.Embedding(words_size, embedding_size)(self._inputs)

        # 对每个球的位置分别处理
        first_lstm = tf.convert_to_tensor(
            [tf.keras.layers.LSTM(hidden_size)(embedding[:, :, i, :]) for i in range(ball_num)]
        )
        first_lstm = tf.transpose(first_lstm, perm=[1, 0, 2])

        # 多层LSTM
        second_lstm = None
        for _ in range(layer_size):
            second_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(first_lstm)

        self._outputs = tf.keras.layers.Dense(n_class)(second_lstm)

        # 构建损失函数
        self._log_likelihood, self._transition_params = crf_log_likelihood(
            self._outputs, self._tag_indices, self._sequence_length
        )
        self._loss = tf.reduce_sum(-self._log_likelihood)

        # 构建预测
        self._pred_sequence, self._viterbi_score = crf_decode(
            self._outputs, self._transition_params, self._sequence_length
        )

        # 添加概率输出
        self._probabilities = tf.nn.softmax(self._outputs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def outputs(self):
        return self._outputs

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def loss(self):
        return self._loss

    @property
    def pred_sequence(self):
        return self._pred_sequence

    @property
    def probabilities(self):
        return self._probabilities


class SignalLstmModel:
    """单向lstm序列模型"""
    
    def __init__(self, batch_size, n_class, w_size, embedding_size, hidden_size, outputs_size, layer_size):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size,), batch_size=batch_size, dtype=tf.int32, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(n_class,), batch_size=batch_size, dtype=tf.float32, name="tag_indices"
        )

        embedding = tf.keras.layers.Embedding(outputs_size, embedding_size)(self._inputs)
        lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(embedding)

        # 多层LSTM
        for _ in range(layer_size):
            lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(lstm)

        final_lstm = tf.keras.layers.LSTM(hidden_size, recurrent_dropout=0.2)(lstm)
        self._outputs = tf.keras.layers.Dense(outputs_size, activation="softmax")(final_lstm)

        # 构建损失函数
        self._loss = - tf.reduce_sum(self._tag_indices * tf.math.log(self._outputs))

        # 预测结果
        self._pred_label = tf.argmax(self.outputs, axis=1)

        # 添加概率输出
        self._probabilities = self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def outputs(self):
        return self._outputs

    @property
    def loss(self):
        return self._loss

    @property
    def pred_label(self):
        return self._pred_label

    @property
    def probabilities(self):
        return self._probabilities