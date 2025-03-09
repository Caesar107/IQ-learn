import torch
import numpy as np

file_path = "experts/CartPole-v1.npy"

# 加载 `Transitions`
transitions = torch.load(file_path)

# 打印 `infos` 的数据类型和前 5 条数据
print("✅ `infos` 数据类型:", type(transitions.infos))
print("✅ `infos` 形状:", transitions.infos.shape)
print("✅ `infos` 示例数据:", transitions.infos[:5])  # 取前 5 条数据

# 检查 `infos` 里每个元素的数据类型
if isinstance(transitions.infos, np.ndarray):
    print("✅ `infos` 里的元素数据类型:", type(transitions.infos[0]))
print("✅ 是否有 rewards:", hasattr(transitions, "rewards"))
