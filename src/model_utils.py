import torch
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将输入转换为 NumPy 数组
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()  # 转换 GPU 张量为 NumPy
    return np.asarray(x)

def check_tensor(X: torch.Tensor):
    """
    Ensures the input is a tensor and moves it to the correct device.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Input data to check and convert if necessary.

    Returns
    -------
    torch.Tensor
        Tensor moved to the specified device.

    Notes
    -----
    This method automatically handles input conversion from numpy arrays to tensors.
    """
    if isinstance(X, torch.Tensor):
        return X.to(DEVICE)
    else:
        return torch.from_numpy(np.asarray(X)).to(DEVICE)