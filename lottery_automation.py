# -*- coding: utf-8 -*-
"""
彩票模型训练和预测自动化脚本
作者：根据需求编写
功能：自动执行数据获取、模型训练、预测和结果记录的循环流程
增强版本：支持命令行参数自定义初始epochs和循环轮数，跨天运行时续写同一文件
修复版本：修复预测结果无法正确保存的问题，增加实时输出显示
"""

import os
import sys
import subprocess
import re
import time
import datetime
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import locale
import shutil
import fnmatch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LotteryAutomation:
    def __init__(self, red_epochs=130, blue_epochs=120, total_rounds=20, work_dir=r"C:\plt5"):
        self.work_dir = work_dir
        self.env_name = "dlt"

        # 使用传入的参数
        self.initial_red_epochs = red_epochs
        self.initial_blue_epochs = blue_epochs
        self.total_rounds = total_rounds

        # 程序启动时间（用于确定文件名）
        self.start_time = datetime.datetime.now()
        self.start_date = self.start_time.strftime("%Y%m%d")

        # 结果文件名使用启动日期，确保跨天运行时使用同一文件
        self.result_file = f"dlt_{self.start_date}_r{red_epochs}b{blue_epochs}_n{total_rounds}.txt"

        # 获取系统编码
        try:
            self.system_encoding = locale.getpreferredencoding()
        except:
            self.system_encoding = 'gbk'  # 默认使用gbk编码

        logger.info(f"系统编码: {self.system_encoding}")
        logger.info(f"程序启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(
            f"初始化参数 - 红球epochs: {self.initial_red_epochs}, 蓝球epochs: {self.initial_blue_epochs}, 总轮数: {self.total_rounds}")

        # 确保结果文件存在
        self.ensure_result_file()

    def ensure_result_file(self):
        """确保结果文件存在，如果不存在则创建"""
        result_path = os.path.join(self.work_dir, self.result_file)

        # 检查文件是否已存在
        file_exists = os.path.exists(result_path)

        try:
            mode = 'a' if file_exists else 'w'  # 如果文件存在则追加，否则创建

            with open(result_path, mode, encoding='utf-8') as f:
                if not file_exists:
                    # 新文件，写入头部
                    f.write(f"DLT训练预测结果汇总 - 程序启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"初始参数: 红球epochs={self.initial_red_epochs}, 蓝球epochs={self.initial_blue_epochs}\n")
                    f.write(f"总轮数: {self.total_rounds}\n")
                    f.write("=" * 60 + "\n\n")
                    logger.info(f"创建结果文件: {result_path}")
                else:
                    # 文件已存在，添加续写标识
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n\n{'=' * 60}\n")
                    f.write(f"程序继续执行 - 时间: {current_time}\n")
                    f.write(f"当前参数: 红球epochs={self.initial_red_epochs}, 蓝球epochs={self.initial_blue_epochs}\n")
                    f.write(f"剩余轮数: {self.total_rounds}\n")
                    f.write("=" * 60 + "\n\n")
                    logger.info(f"续写现有结果文件: {result_path}")

        except Exception as e:
            logger.error(f"处理结果文件失败: {e}")

    def get_current_time_header(self):
        """获取当前时间头部，用于标记每轮执行的时间"""
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y-%m-%d %H:%M:%S")

    def run_command(self, command, description=""):
        """运行命令并实时显示输出 - 修复编码问题"""
        logger.info(f"执行命令: {description}")
        logger.info(f"命令: {command}")

        try:
            # 切换到工作目录
            original_dir = os.getcwd()
            os.chdir(self.work_dir)

            # 使用正确的编码执行命令
            if sys.platform.startswith('win'):
                # Windows系统使用cmd
                full_command = f'cmd /c "conda activate {self.env_name} && {command}"'
            else:
                # Linux/Mac使用bash
                full_command = f'bash -c "conda activate {self.env_name} && {command}"'

            # 创建子进程并实时捕获输出
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                bufsize=1,  # 行缓冲
                universal_newlines=True,  # 文本模式
                encoding='utf-8',
                errors='replace'  # 替换编码错误
            )

            # 实时读取并显示输出
            output_lines = []
            print(f"\n=== {description} 开始执行 ===")

            while True:
                # 实时读取一行输出
                output = process.stdout.readline()

                if output == '' and process.poll() is not None:
                    break

                if output:
                    # 清理输出中的特殊字符和多余空白
                    cleaned_output = output.strip()
                    if cleaned_output:
                        # 实时显示到控制台
                        print(f"{description}: {cleaned_output}")

                        # 同时记录到日志
                        logger.info(f"{description}输出: {cleaned_output}")

                        # 保存到输出列表
                        output_lines.append(cleaned_output)

            # 获取剩余输出
            remaining_output, _ = process.communicate()
            if remaining_output:
                cleaned_remaining = remaining_output.strip()
                if cleaned_remaining:
                    print(f"{description}: {cleaned_remaining}")
                    logger.info(f"{description}输出: {cleaned_remaining}")
                    output_lines.append(cleaned_remaining)

            # 切换回原目录
            os.chdir(original_dir)

            # 检查返回码
            return_code = process.poll()

            if return_code == 0:
                logger.info(f"{description} 执行成功")
                print(f"=== {description} 执行成功 ===\n")
                return True, '\n'.join(output_lines)
            else:
                logger.error(f"{description} 执行失败，返回码: {return_code}")
                error_msg = '\n'.join(output_lines) if output_lines else "未知错误"
                logger.error(f"错误信息: {error_msg}")
                print(f"=== {description} 执行失败，返回码: {return_code} ===\n")
                return False, error_msg

        except subprocess.TimeoutExpired:
            logger.error(f"{description} 执行超时")
            print(f"=== {description} 执行超时 ===\n")
            # 尝试终止进程
            try:
                process.kill()
            except:
                pass
            return False, "执行超时"
        except Exception as e:
            logger.error(f"执行命令时发生异常: {e}")
            print(f"=== {description} 执行异常: {e} ===\n")
            return False, str(e)



    def get_config_params(self):
        """获取config.py中的当前参数值"""
        config_path = os.path.join(self.work_dir, "config.py")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式匹配参数
            red_epochs_match = re.search(r'"red_epochs":\s*(\d+)', content)
            blue_epochs_match = re.search(r'"blue_epochs":\s*(\d+)', content)

            if red_epochs_match and blue_epochs_match:
                red_epochs = int(red_epochs_match.group(1))
                blue_epochs = int(blue_epochs_match.group(1))
                logger.info(f"当前参数 - red_epochs: {red_epochs}, blue_epochs: {blue_epochs}")
                return red_epochs, blue_epochs
            else:
                logger.error("无法从config.py中读取参数")
                return None, None

        except Exception as e:
            logger.error(f"读取config.py时发生错误: {e}")
            return None, None

    def update_config_params(self, red_epochs, blue_epochs):
        """更新config.py中的参数值"""
        config_path = os.path.join(self.work_dir, "config.py")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换参数值
            content = re.sub(r'"red_epochs":\s*\d+', f'"red_epochs": {red_epochs}', content)
            content = re.sub(r'"blue_epochs":\s*\d+', f'"blue_epochs": {blue_epochs}', content)

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"参数已更新 - red_epochs: {red_epochs}, blue_epochs: {blue_epochs}")
            return True

        except Exception as e:
            logger.error(f"更新config.py时发生错误: {e}")
            return False

    def ensure_target_params(self):
        """确保参数值为目标值（使用传入的初始值）"""
        current_red, current_blue = self.get_config_params()

        if current_red is None or current_blue is None:
            logger.error("无法获取当前参数，使用默认值")
            return self.update_config_params(self.initial_red_epochs, self.initial_blue_epochs)

        # 修复：确保参数在合理范围内
        if current_red < 10 or current_red > 1000 or current_blue < 10 or current_blue > 1000:
            logger.info("参数超出合理范围，重置为初始值")
            return self.update_config_params(self.initial_red_epochs, self.initial_blue_epochs)

        if current_red != self.initial_red_epochs or current_blue != self.initial_blue_epochs:
            logger.info(f"参数不符合要求，正在调整为 red_epochs={self.initial_red_epochs}, blue_epochs={self.initial_blue_epochs}")
            return self.update_config_params(self.initial_red_epochs, self.initial_blue_epochs)
        else:
            logger.info("参数已符合要求")
            return True

    def write_result_header(self, red_epochs, blue_epochs, phase, round_num):
        """写入结果文件头部 - 添加当前时间标记"""
        result_path = os.path.join(self.work_dir, self.result_file)

        try:
            with open(result_path, 'a', encoding='utf-8') as f:
                current_time = self.get_current_time_header()
                if phase == "train":
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"第{round_num}轮训练结果 - red_epochs: {red_epochs}, blue_epochs: {blue_epochs}\n")
                    f.write(f"执行时间: {current_time}\n")
                    f.write(f"{'=' * 60}\n\n")
                elif phase == "predict":
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"第{round_num}轮预测结果 - red_epochs: {red_epochs}, blue_epochs: {blue_epochs}\n")
                    f.write(f"执行时间: {current_time}\n")
                    f.write(f"{'=' * 60}\n\n")

            logger.info(f"已写入第{round_num}轮{phase}结果头部")
            return True

        except Exception as e:
            logger.error(f"写入结果文件头部时发生错误: {e}")
            return False

    def _process_training_output(self, output, round_num):
        """处理训练输出，提取并显示关键信息"""
        if not output:
            return

        print(f"\n第{round_num}轮训练关键信息:")
        print("-" * 50)

        lines = output.split('\n')
        key_phrases = [
            "epoch", "loss", "红球", "蓝球", "命中", "AC值", "断区",
            "测试期数", "CRF预测", "复式预测", "准确率", "耗时"
        ]

        for line in lines:
            if any(phrase in line.lower() for phrase in key_phrases):
                print(f"训练输出: {line.strip()}")

    def _process_prediction_output(self, output, round_num):
        """处理预测输出，提取并显示关键信息 - 增强过滤版本"""
        if not output:
            return

        print(f"\n第{round_num}轮预测关键信息:")
        print("-" * 50)

        lines = output.split('\n')

        # 定义日志关键词，用于过滤
        log_keywords = ['INFO', 'WARNING', 'ERROR', 'tensorflow', 'INFO |', 'WARN', 'DEBUG']

        # 定义预测关键词
        pred_keywords = [
            "期号", "预测", "红球", "蓝球", "Top", "AC值", "断区",
            "CRF", "复式", "得分", "排名", "增强预测"
        ]

        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                # 过滤掉日志行，只显示预测结果
                if (not any(keyword in line_stripped for keyword in log_keywords) and
                        any(phrase in line_stripped for phrase in pred_keywords)):
                    print(f"预测输出: {line_stripped}")

    def cleanup_previous_prediction_files(self, round_num):
        """清理上一轮的预测文件，避免重复记录"""
        try:
            # 要删除的文件模式
            patterns = [
                "dlt_enhanced.txt",
                "dlt_enhanced_*.txt",
                "dlt_prediction_*.txt"
            ]

            deleted_files = []
            for pattern in patterns:
                for file in os.listdir(self.work_dir):
                    if fnmatch.fnmatch(file, pattern):
                        file_path = os.path.join(self.work_dir, file)
                        try:
                            os.remove(file_path)
                            deleted_files.append(file)
                            logger.info(f"已删除预测文件: {file}")
                        except Exception as e:
                            logger.warning(f"删除文件 {file} 失败: {e}")

            if deleted_files:
                logger.info(f"已清理 {len(deleted_files)} 个预测文件: {', '.join(deleted_files)}")
            else:
                logger.info("未找到需要清理的预测文件")

        except Exception as e:
            logger.error(f"清理预测文件时发生错误: {e}")

    def run_train_model(self, red_epochs, blue_epochs, round_num):
        """运行模型训练脚本 - 修复版本：在训练前删除旧的预测文件"""

        # 在新一轮训练开始前删除旧的预测文件
        self.cleanup_previous_prediction_files(round_num)

        # 先写入结果头部
        self.write_result_header(red_epochs, blue_epochs, "train", round_num)

        command = f"python run_train_model.py --name {self.env_name} --train_test_split 0.65"

        # 修复：去掉不存在的save_to_file参数
        success, output = self.run_command(command, "模型训练")

        if success:
            # 处理训练输出，显示关键信息
            self._process_training_output(output, round_num)

            # 即使不保存过程输出，仍然保存训练脚本生成的单独结果文件
            self.save_separate_results("train", round_num)

            logger.info(f"第{round_num}轮训练执行成功")
        else:
            logger.error(f"第{round_num}轮训练执行失败")

        return success

    def run_predict(self, red_epochs, blue_epochs, round_num):
        """运行预测脚本 - 修复版本：只保存关键结果，不保存命令输出"""
        # 先写入结果头部
        self.write_result_header(red_epochs, blue_epochs, "predict", round_num)

        # 删除可能存在的旧预测文件，避免混淆
        enhanced_file = os.path.join(self.work_dir, "dlt_enhanced.txt")
        if os.path.exists(enhanced_file):
            try:
                # 备份而不是删除，以便后续查找
                backup_file = os.path.join(self.work_dir, f"dlt_enhanced_backup_round_{round_num}.txt")
                shutil.copy2(enhanced_file, backup_file)
                logger.info(f"已备份预测文件: {backup_file}")
            except Exception as e:
                logger.warning(f"备份预测文件失败: {e}")

        command = f"python run_predict.py --name {self.env_name}"
        success, output = self.run_command(command, "模型预测")

        if success:
            # 额外处理预测输出，显示关键信息
            self._process_prediction_output(output, round_num)

            # 直接保存关键预测结果，不保存命令输出
            try:
                self.save_separate_results("predict", round_num)
                logger.info(f"第{round_num}轮预测关键结果已保存")
            except Exception as e:
                logger.error(f"保存预测结果时发生错误: {e}")

                # 如果保存失败，只写入简化的成功信息
                result_path = os.path.join(self.work_dir, self.result_file)
                try:
                    with open(result_path, 'a', encoding='utf-8') as f:
                        f.write(f"第{round_num}极预测执行成功\n")
                except Exception as e2:
                    logger.error(f"写入简化结果失败: {e2}")
        else:
            logger.error(f"第{round_num}轮预测执行失败")

            # 失败时只写入简化的错误信息
            result_path = os.path.join(self.work_dir, self.result_file)
            try:
                with open(result_path, 'a', encoding='utf-8') as f:
                    f.write(f"第{round_num}轮预测执行失败\n")
                    # 只写入错误摘要，不写完整输出
                    if output and len(output) > 200:
                        f.write(f"错误摘要: {output[:200]}...\n")
                    elif output:
                        f.write(f"错误信息: {output}\n")
            except Exception as e:
                logger.error(f"保存预测错误信息失败: {e}")

        return success


    def find_training_summary_file(self, round_num):
        """改进的训练总结文件查找逻辑"""
        try:
            # 多种可能的文件名模式
            patterns = [
                f"dlt_training_summary_*.txt",  # 带时间戳的文件
                f"dlt_training_summary.txt",  # 不带时间戳的文件
                f"training_summary_round_{round_num}.txt",  # 按轮次命名的文件
                f"dlt_round_{round_num:02d}_train_result.txt"  # 统一格式的文件
            ]

            for pattern in patterns:
                matching_files = []
                for file in os.listdir(self.work_dir):
                    if fnmatch.fnmatch(file, pattern):
                        matching_files.append(file)

                if matching_files:
                    # 按修改时间排序，取最新的文件
                    matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.work_dir, x)), reverse=True)
                    return os.path.join(self.work_dir, matching_files[0])

            # 如果没找到，尝试在logs目录中查找
            logs_dir = os.path.join(self.work_dir, "logs")
            if os.path.exists(logs_dir):
                for pattern in patterns:
                    for file in os.listdir(logs_dir):
                        if fnmatch.fnmatch(file, pattern):
                            return os.path.join(logs_dir, file)

            logger.warning(f"第{round_num}轮训练总结文件未找到")
            return None

        except Exception as e:
            logger.error(f"查找训练总结文件时出错: {e}")
            return None

    def find_prediction_file(self, round_num):
        """改进的预测文件查找逻辑"""
        try:
            # 优先查找增强预测文件
            possible_files = [
                "dlt_enhanced.txt",  # 主要预测文件
                f"dlt_enhanced_round_{round_num}.txt",
                f"dlt_prediction_{round_num}.txt",
                f"dlt_round_{round_num:02d}_predict.txt"
            ]

            for filename in possible_files:
                file_path = os.path.join(self.work_dir, filename)
                if os.path.exists(file_path):
                    logger.info(f"找到预测文件: {file_path}")
                    return file_path

            # 如果没找到，检查工作目录下所有相关文件
            for file in os.listdir(self.work_dir):
                if file.startswith("dlt_enhanced") or file.startswith("dlt_prediction"):
                    file_path = os.path.join(self.work_dir, file)
                    logger.info(f"找到预测相关文件: {file_path}")
                    return file_path

            logger.warning(f"第{round_num}轮预测文件未找到")
            return None

        except Exception as e:
            logger.error(f"查找预测文件时出错: {e}")
            return None

    def save_separate_results(self, phase, round_num):
        """只保存关键结果，不保存过程输出"""
        try:
            result_path = os.path.join(self.work_dir, self.result_file)

            with open(result_path, 'a', encoding='utf-8') as result_file:
                if phase == "train":
                    training_file = self.find_training_summary_file(round_num)

                    if training_file and os.path.exists(training_file):
                        try:
                            with open(training_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()

                            if content:
                                # 只提取关键结果信息
                                key_results = []
                                lines = content.split('\n')

                                # 定义关键信息标识
                                key_phrases = [
                                    '准确率', 'AC值', 'CRF预测', '复式预测',
                                    '测试期数', '平均损失', '耗时', 'epoch',
                                    'loss', '命中', '断区', '综合得分'
                                ]

                                # 提取包含关键信息的行
                                for line in lines:
                                    line_lower = line.lower()
                                    if any(phrase in line_lower for phrase in key_phrases):
                                        key_results.append(line.strip())

                                if key_results:
                                    result_file.write(f"\n第{round_num}轮训练关键结果:\n")
                                    result_file.write("-" * 40 + "\n")
                                    result_file.write('\n'.join(key_results) + "\n")
                                    logger.info(f"第{round_num}轮训练关键结果已保存")
                                else:
                                    # 如果没有找到关键信息，保存文件的前几行和后几行
                                    if len(lines) > 10:
                                        summary_lines = lines[:5] + ["..."] + lines[-5:]
                                    else:
                                        summary_lines = lines

                                    result_file.write(f"\n第{round_num}轮训练结果摘要:\n")
                                    result_file.write("-" * 40 + "\n")
                                    result_file.write('\n'.join(summary_lines) + "\n")
                                    logger.info(f"第{round_num}轮训练结果摘要已保存")

                        except Exception as e:
                            logger.error(f"读取训练文件失败: {e}")
                            result_file.write(f"\n第{round_num}轮训练结果:\n")
                            result_file.write("-" * 40 + "\n")
                            result_file.write(f"读取训练文件失败: {e}\n")
                    else:
                        # 创建占位符，确保轮次完整性
                        result_file.write(f"\n第{round_num}轮训练结果:\n")
                        result_file.write("-" * 40 + "\n")
                        result_file.write("训练结果文件未找到\n")
                        logger.warning(f"第{round_num}轮训练结果文件不存在")

                elif phase == "predict":
                    prediction_file = self.find_prediction_file(round_num)

                    if prediction_file and os.path.exists(prediction_file):
                        try:
                            with open(prediction_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read().strip()

                            if content:
                                # 只提取关键预测结果
                                key_predictions = []
                                lines = content.split('\n')

                                # 定义预测关键信息标识
                                pred_key_phrases = [
                                    '期号', '预测', '红球', '蓝球', 'Top', 'AC值',
                                    '断区', 'CRF', '复式', '得分', '排名', '增强预测'
                                ]

                                # 查找包含预测结果的块
                                in_prediction_block = False
                                prediction_block = []

                                for line in lines:
                                    line_stripped = line.strip()
                                    if not line_stripped:
                                        continue

                                    # 检查是否进入预测结果块
                                    if any(phrase in line_stripped for phrase in ['期号', '预测时间', '大乐透']):
                                        in_prediction_block = True
                                        prediction_block = [line_stripped]
                                    elif in_prediction_block:
                                        if line_stripped.startswith('===') or line_stripped.startswith('---'):
                                            # 块结束标记
                                            if prediction_block:
                                                key_predictions.extend(prediction_block)
                                                key_predictions.append("")  # 空行分隔
                                            in_prediction_block = False
                                            prediction_block = []
                                        else:
                                            prediction_block.append(line_stripped)
                                    elif any(phrase in line_stripped.lower() for phrase in pred_key_phrases):
                                        key_predictions.append(line_stripped)

                                # 如果没找到结构化块，取文件的开头和结尾部分
                                if not key_predictions and len(lines) > 0:
                                    if len(lines) > 15:
                                        key_predictions = lines[:8] + ["..."] + lines[-7:]
                                    else:
                                        key_predictions = lines

                                if key_predictions:
                                    result_file.write(f"\n第{round_num}轮预测关键结果:\n")
                                    result_file.write("-" * 40 + "\n")
                                    result_file.write('\n'.join(key_predictions) + "\n\n")
                                    logger.info(f"第{round_num}轮预测关键结果已保存")
                                else:
                                    logger.warning(f"预测文件内容为空或格式异常: {prediction_file}")
                                    result_file.write(f"\n第{round_num}轮预测结果:\n")
                                    result_file.write("-" * 40 + "\n")
                                    result_file.write("预测结果文件内容为空或格式异常\n\n")

                        except Exception as e:
                            logger.error(f"读取预测文件失败: {e}")
                            result_file.write(f"\n第{round_num}轮预测结果:\n")
                            result_file.write("-" * 40 + "\n")
                            result_file.write(f"读取预测文件失败: {e}\n\n")
                    else:
                        # 创建占位符，确保轮次完整性
                        result_file.write(f"\n第{round_num}轮预测结果:\n")
                        result_file.write("-" * 40 + "\n")
                        result_file.write("预测结果文件未找到\n\n")
                        logger.warning(f"第{round_num}轮预测结果文件不存在")

            logger.info(f"第{round_num}轮{phase}关键结果已添加到汇总文件")

        except Exception as e:
            logger.error(f"保存{phase}结果时发生错误: {e}")


    def check_round_completeness(self):
        """检查轮次完整性 - 修复版本"""
        try:
            result_path = os.path.join(self.work_dir, self.result_file)

            if not os.path.exists(result_path):
                logger.error("结果文件不存在")
                return []

            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()

            missing_rounds = []

            for round_num in range(1, self.total_rounds + 1):
                # 检查训练结果标记
                train_pattern1 = f"第{round_num}轮训练详细结果:"
                train_pattern2 = f"第{round_num}轮训练结果"

                # 检查预测结果标记
                predict_pattern1 = f"第{round_num}轮预测详细结果:"
                predict_pattern2 = f"第{round_num}轮预测结果"

                train_found = train_pattern1 in content or train_pattern2 in content
                predict_found = predict_pattern1 in content or predict_pattern2 in content

                if not train_found:
                    missing_rounds.append(f"第{round_num}轮训练结果")
                if not predict_found:
                    missing_rounds.append(f"第{round_num}轮预测结果")

            if missing_rounds:
                logger.warning(f"发现缺失的轮次: {missing_rounds}")
            else:
                logger.info("所有轮次结果完整")

            return missing_rounds

        except Exception as e:
            logger.error(f"检查轮次完整性时出错: {e}")
            return []

    def send_email(self):
        """发送邮件通知"""
        result_path = os.path.join(self.work_dir, self.result_file)

        if not os.path.exists(result_path):
            logger.error("结果文件不存在，无法发送邮件")
            return False

        try:
            # 读取结果文件内容
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 邮件配置（需要替换为实际配置）
            smtp_server = "smtp.qq.com"
            smtp_port = 587
            sender_email = "424077344@qq.com"
            receiver_email = "402591270@qq.com"
            password = "xorqiogqxzprbhjh"

            # 创建邮件
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = f"DLT训练预测结果 (启动于{self.start_date})"

            # 邮件正文
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            hours = duration.total_seconds() / 3600

            body = f"DLT训练预测结果汇总\n"
            body += f"程序启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            body += f"程序结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            body += f"运行时长: {hours:.2f} 小时\n"
            body += f"初始参数: 红球epochs={self.initial_red_epochs}, 蓝球epochs={self.initial_blue_epochs}\n"
            body += f"总轮数: {self.total_rounds}\n\n"
            body += "详细结果请查看附件。"
            message.attach(MIMEText(body, "plain"))

            # 添加附件
            with open(result_path, 'r', encoding='utf-8') as f:
                attachment = MIMEText(f.read(), _subtype='plain', _charset='utf-8')
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(result_path))
                message.attach(attachment)

            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(message)

            logger.info("邮件发送成功")
            return True

        except Exception as e:
            logger.error(f"发送邮件时发生错误: {e}")
            return False

    def run_get_data(self):
        """运行数据获取脚本"""
        command = f"python get_data.py --name {self.env_name}"
        success, output = self.run_command(command, "数据获取")
        return success

    def run_automation(self):
        """改进的自动化流程执行方法 - 增强进度显示"""
        logger.info("开始执行DLT自动化流程")
        print("\n" + "=" * 60)
        print("DLT自动化流程启动")
        print(f"启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"初始参数: 红球epochs={self.initial_red_epochs}, 蓝球epochs={self.initial_blue_epochs}")
        print(f"总轮数: {self.total_rounds}")
        print("=" * 60 + "\n")

        try:
            # 步骤1: 运行数据获取
            print("步骤1: 运行数据获取")
            logger.info("步骤1: 运行数据获取")
            success = self.run_get_data()
            if not success:
                logger.error("数据获取失败，终止流程")
                print("数据获取失败，终止流程")
                return False

            # 步骤2: 检查并设置参数
            print("步骤2: 检查并设置参数")
            logger.info("步骤2: 检查并设置参数")
            if not self.ensure_target_params():
                logger.error("参数设置失败，终止流程")
                print("参数设置失败，终止流程")
                return False

            # 步骤3: 执行指定轮数的循环
            print(f"步骤3: 开始{self.total_rounds}轮循环")
            logger.info(f"步骤3: 开始{self.total_rounds}轮循环")

            completed_rounds = 0

            for round_num in range(1, self.total_rounds + 1):
                # 检查当前日期，如果跨天则记录日志
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                if current_date != self.start_date:
                    logger.info(f"注意：程序已跨天运行，当前日期: {current_date}，启动日期: {self.start_date}")
                    print(f"注意：程序已跨天运行，当前日期: {current_date}")

                print(f"\n{'=' * 40}")
                print(f"开始第{round_num}轮执行")
                print(f"{'=' * 40}")
                logger.info(f"第{round_num}轮开始")

                # 获取当前参数
                current_red, current_blue = self.get_config_params()
                if current_red is None or current_blue is None:
                    logger.error("无法获取当前参数，跳过本轮")
                    print("无法获取当前参数，跳过本轮")
                    continue

                print(f"当前参数: red_epochs={current_red}, blue_epochs={current_blue}")
                logger.info(f"第{round_num}轮参数: red_epochs={current_red}, blue_epochs={current_blue}")

                # 运行训练
                print(f"\n第{round_num}轮 - 开始模型训练")
                train_success = self.run_train_model(current_red, current_blue, round_num)

                # 运行预测
                print(f"\n第{round_num}轮 - 开始模型预测")
                predict_success = self.run_predict(current_red, current_blue, round_num)

                # 更新参数（+1）
                new_red = current_red + 1
                new_blue = current_blue + 1
                if not self.update_config_params(new_red, new_blue):
                    logger.error(f"第{round_num}轮参数更新失败")
                    print(f"第{round_num}轮参数更新失败")
                else:
                    logger.info(f"参数已更新 - red_epochs: {new_red}, blue_epochs: {new_blue}")
                    print(f"参数已更新: red_epochs={new_red}, blue_epochs={new_blue}")

                if train_success and predict_success:
                    completed_rounds += 1
                    logger.info(f"第{round_num}轮完成")
                    print(f"第{round_num}轮完成 ✓")
                else:
                    logger.warning(f"第{round_num}轮部分任务失败")
                    print(f"第{round_num}轮部分任务失败 ⚠")

                # 短暂暂停，避免过于密集的执行
                time.sleep(5)

            # 步骤4: 检查轮次完整性
            print(f"\n步骤4: 检查轮次完整性")
            logger.info("步骤4: 检查轮次完整性")
            missing_rounds = self.check_round_completeness()

            if missing_rounds:
                logger.warning(f"完成{completed_rounds}/{self.total_rounds}轮，缺失轮次: {len(missing_rounds)}")
                print(f"完成{completed_rounds}/{self.total_rounds}轮，缺失轮次: {len(missing_rounds)}")
            else:
                logger.info(f"成功完成{completed_rounds}/{self.total_rounds}轮")
                print(f"成功完成{completed_rounds}/{self.total_rounds}轮 ✓")

            # 步骤5: 发送邮件
            print(f"\n步骤5: 发送邮件通知")
            logger.info("步骤5: 发送邮件")
            if not self.send_email():
                logger.error("邮件发送失败")
                print("邮件发送失败 ⚠")
                return False

            print(f"\n{'=' * 60}")
            print("自动化流程完成")
            print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总运行时长: {(datetime.datetime.now() - self.start_time).total_seconds() / 60:.2f} 分钟")
            print(f"{'=' * 60}\n")

            logger.info("自动化流程完成")
            return True

        except Exception as e:
            logger.error(f"自动化流程执行过程中发生异常: {e}")
            print(f"自动化流程执行过程中发生异常: {e}")
            return False


def main():
    """主函数 - 添加命令行参数解析"""
    parser = argparse.ArgumentParser(description='DLT彩票模型自动化训练和预测')
    parser.add_argument('--red_epochs', type=int, default=130,
                        help='初始红球训练轮数 (默认: 130)')
    parser.add_argument('--blue_epochs', type=int, default=120,
                        help='初始蓝球训练轮数 (默认: 120)')
    parser.add_argument('--rounds', type=int, default=20,
                        help='总循环轮数 (默认: 20)')
    parser.add_argument('--work_dir', type=str, default=r"C:\plt5",
                        help='工作目录路径 (默认: C:\plt5)')

    args = parser.parse_args()

    # 显示参数信息
    print("=" * 60)
    print("DLT彩票模型自动化脚本")
    print("=" * 60)
    print(f"初始红球epochs: {args.red_epochs}")
    print(f"初始蓝球epochs: {args.blue_epochs}")
    print(f"总循环轮数: {args.rounds}")
    print(f"工作目录: {args.work_dir}")
    print("=" * 60)
    print("开始执行...")

    # 创建自动化实例
    automation = LotteryAutomation(
        red_epochs=args.red_epochs,
        blue_epochs=args.blue_epochs,
        total_rounds=args.rounds,
        work_dir=args.work_dir
    )

    try:
        success = automation.run_automation()
        if success:
            logger.info("自动化流程执行成功")
            print("自动化流程执行成功！")
        else:
            logger.error("自动化流程执行失败")
            print("自动化流程执行失败，请查看日志文件了解详情。")

    except Exception as e:
        logger.error(f"自动化流程执行过程中发生异常: {e}")
        print(f"程序执行过程中发生异常: {e}")


if __name__ == "__main__":
    main()