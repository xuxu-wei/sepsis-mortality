import torch
import numpy as np
# 将输入转换为 NumPy 数组
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()  # 转换 GPU 张量为 NumPy
    return np.asarray(x)