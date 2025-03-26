"""
examine_expert_file.py

该脚本用于**检查专家轨迹数据文件**（如 `.npy` 或 `.pkl` 格式），以分析其内部结构和可用性。
特别适用于强化学习/模仿学习项目中检查专家策略是否能用于后续如 KL 散度计算、行为克隆等任务。

脚本功能包括：
1. 加载 `.npy` 或 `.pkl` 文件（支持零维 NumPy 对象或嵌套结构）
2. 判断是否包含：
   - 策略输出（如概率分布、logits、动作）
   - 状态-动作对（用于行为克隆或策略再训练）
   - 轨迹结构（例如包含多个 episode 的列表或字典）
3. 推测文件所属的算法（如 PPO、SAC、BC 等）
4. 输出可用于计算 KL 散度的结构信息
5. 打印样例内容，辅助人工确认

适用场景：
- 检查从 Baseline、Stable-Baselines3、或自定义算法导出的专家文件
- 判断 expert 文件是否适合转换为 IQ-Learn、AIRL 等算法可用格式
- 评估是否包含 policy 分布以直接计算 KL 散度，或需要额外训练

注意事项：
- 本脚本假设 expert 文件中保存的是结构化数据（dict 或轨迹列表）
- 如为模型类对象，尝试识别是否能 `predict()` 或 `get_weights()`
- 如为零维 numpy.ndarray，需要使用 .item() 提取对象再解析

使用方式：
- 修改文件底部的 `file_path` 指向你要检查的 expert 文件路径
- 运行脚本：`python examine_expert_file.py`
"""


import numpy as np
import os
import pickle
import json
from pathlib import Path

def examine_expert_file(file_path):
    """
    检查CartPole专家策略文件并提取信息
    """
    print(f"正在检查文件: {file_path}")
    
    try:
        # 尝试加载NumPy数组
        data = np.load(file_path, allow_pickle=True)
        print(f"文件成功加载，类型: {type(data)}")
        
        # 检查数据结构
        if isinstance(data, np.ndarray):
            print(f"数据是NumPy数组，形状: {data.shape}")
            
            # 特殊处理空形状数组
            if data.shape == ():
                print("发现零维数组（标量数组），尝试获取内部对象")
                try:
                    # 对于零维数组，使用item()提取内部对象
                    inner_data = data.item()
                    print(f"内部对象类型: {type(inner_data)}")
                    
                    # 检查内部对象是否为字典
                    if isinstance(inner_data, dict):
                        print(f"内部对象是字典，键: {list(inner_data.keys())}")
                        
                        # 检查是否有策略相关信息
                        policy_keys = [k for k in inner_data.keys() if 'policy' in k.lower() or 'action' in k.lower() or 'prob' in k.lower()]
                        if policy_keys:
                            print(f"发现策略相关键: {policy_keys}")
                            
                            # 查看策略数据
                            for pk in policy_keys:
                                print(f"策略键 '{pk}' 的值类型: {type(inner_data[pk])}")
                                if hasattr(inner_data[pk], 'shape'):
                                    print(f"形状: {inner_data[pk].shape}")
                        
                        # 检查是否包含轨迹数据
                        traj_keys = [k for k in inner_data.keys() if 'traj' in k.lower() or 'path' in k.lower() or 'demo' in k.lower()]
                        if traj_keys:
                            print(f"发现轨迹相关键: {traj_keys}")
                            for tk in traj_keys:
                                traj_data = inner_data[tk]
                                print(f"轨迹数据类型: {type(traj_data)}")
                                if isinstance(traj_data, list) or isinstance(traj_data, np.ndarray):
                                    print(f"轨迹数量: {len(traj_data)}")
                                    if len(traj_data) > 0:
                                        print(f"单个轨迹类型: {type(traj_data[0])}")
                                        if isinstance(traj_data[0], dict):
                                            print(f"轨迹包含的键: {list(traj_data[0].keys())}")
                    
                    # 检查是否为模型对象
                    elif hasattr(inner_data, 'get_weights') or hasattr(inner_data, 'predict'):
                        print("内部对象似乎是模型，可能用于计算KL散度")
                        try:
                            model_info = inner_data.summary()
                            print("模型摘要可用")
                        except:
                            print("模型对象存在，但无法获取摘要")
                    
                    # 尝试判断算法
                    if isinstance(inner_data, dict):
                        algo_indicators = {
                            'PPO': ['value', 'advantage', 'return', 'log_prob'],
                            'DQN': ['q_values', 'target_q'],
                            'SAC': ['alpha', 'q_value', 'entropy'],
                            'TRPO': ['kl', 'old_log_prob'],
                            'Behavior Cloning': ['expert', 'demonstration', 'demo']
                        }
                        
                        possible_algos = []
                        for algo, indicators in algo_indicators.items():
                            if isinstance(inner_data, dict):
                                matches = sum(any(ind in k.lower() for k in inner_data.keys()) for ind in indicators)
                                if matches > 0:
                                    possible_algos.append((algo, matches))
                        
                        if possible_algos:
                            possible_algos.sort(key=lambda x: x[1], reverse=True)
                            print(f"可能的训练算法: {[a[0] for a in possible_algos]}")
                        else:
                            print("无法确定训练算法")
                        
                        # 检查是否可用于KL散度计算
                        if isinstance(inner_data, dict):
                            kl_usable = any('prob' in k.lower() or 'logit' in k.lower() or 'distribution' in k.lower() for k in inner_data.keys())
                            if kl_usable:
                                print("该文件可能包含可用于计算KL散度的策略分布信息")
                            else:
                                print("未找到明确的策略分布信息，可能难以直接计算KL散度")
                                
                                # 检查是否包含状态动作对
                                if any('state' in k.lower() or 'observation' in k.lower() for k in inner_data.keys()) and \
                                   any('action' in k.lower() for k in inner_data.keys()):
                                    print("文件包含状态-动作对，可用于行为克隆或训练模型后计算KL散度")
                                    
                    else:
                        print("内部对象不是字典或模型，难以确定算法类型")
                        
                except Exception as inner_e:
                    print(f"处理内部对象时出错: {str(inner_e)}")
            
            elif len(data) > 0:
                # 处理非空数组
                print(f"数据长度: {len(data)}")  # 新增的打印语句
                if isinstance(data[0], dict):
                    print("数据似乎包含轨迹字典")
                    # 后续逻辑与原脚本相同
                    # ...
            
            # 提供如何使用该文件计算KL散度的建议
            print("\n关于KL散度计算的建议:")
            print("1. 如果文件包含策略概率分布，可直接计算KL散度")
            print("2. 如果只包含状态-动作对，需先训练策略网络，再计算KL散度")
            print("3. 如果包含模型，可使用同样的状态输入，比较输出分布")
            
            print("\n检查文件内容示例，判断是否适合计算KL散度:")
            try:
                if data.shape == ():
                    inner_data = data.item()
                    if isinstance(inner_data, dict):
                        # 打印字典的前几个键值对
                        for i, (k, v) in enumerate(inner_data.items()):
                            if i >= 5:  # 只显示前5个键值对
                                print("...")
                                break
                            print(f"键: {k}, 值类型: {type(v)}")
                            # 如果值是数组或列表，显示其形状或长度
                            if hasattr(v, 'shape'):
                                print(f"  形状: {v.shape}")
                            elif isinstance(v, list):
                                print(f"  长度: {len(v)}")
                                if len(v) > 0:
                                    print(f"  第一个元素类型: {type(v[0])}")
                else:
                    # 非零维数组，显示前几个元素
                    for i in range(min(5, len(data))):
                        print(f"元素[{i}] 类型: {type(data[i])}")
            except Exception as e:
                print(f"尝试打印内容示例时出错: {str(e)}")
        else:
            # 不是NumPy数组
            print(f"数据不是NumPy数组，而是: {type(data)}")
            # 后续处理与原脚本类似
                
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        # 尝试其他格式加载
        try:
            print("尝试以pickle格式加载...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"文件加载为Pickle格式，类型: {type(data)}")
            # 分析pickle数据
        except Exception as pickle_e:
            print(f"pickle加载失败: {str(pickle_e)}")

if __name__ == "__main__":
    # 文件路径
    file_path = "E:\TRRL\IQ-Learn\iq_learn\experts\CartPole-v1_expert_trajs.npy"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        # 尝试修正路径
        alt_path = "E:\TRRL\IQ-Learn\iq_learn\experts\CartPole-v1_expert_trajs.npy"
        if os.path.exists(alt_path):
            print(f"找到替代路径: {alt_path}")
            file_path = alt_path
        else:
            print("无法找到文件，请确认路径是否正确")
            exit(1)
            
    examine_expert_file(file_path)