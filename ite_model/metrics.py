import torch
import numpy as np
import pandas as pd

def RCT_ATE(treatment, y):
    """
    计算随机对照试验 (RCT) 中的 ATE (Average Treatment Effect)。
    
    参数:
    ----------
    treatment : numpy.ndarray, pd.Series, 或 torch.Tensor
        处理组指示变量数组，仅包含 0 和 1。
    y : numpy.ndarray, pd.Series, 或 torch.Tensor
        结果变量数组，与 treatment 的长度相同。
    
    返回:
    ----------
    float
        计算得到的平均处理效应 (ATE)。
    """
    # 检查输入长度是否一致
    if len(treatment) != len(y):
        raise ValueError("treatment 和 y 的长度必须一致")
    
    # 如果输入是 pd.Series，转换为 NumPy 数组
    if isinstance(treatment, pd.Series):
        treatment = treatment.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    # 如果输入是 torch.Tensor，调用 .cpu().numpy()
    if isinstance(treatment, torch.Tensor):
        treatment = treatment.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    # 确保输入为 NumPy 数组
    treatment = np.asarray(treatment)
    y = np.asarray(y)

    # 分别计算处理组和对照组的平均结果
    E_y_1 = y[treatment == 1].mean()  # 平均结果: treatment = 1
    E_y_0 = y[treatment == 0].mean()  # 平均结果: treatment = 0

    # 计算ATE
    ATE = E_y_1 - E_y_0
    return ATE

def RCT_ATE_l1_loss(treatment, y_true, y_pred_0, y_pred_1, eval_strategy='observed_only'):
    """
    计算 ATE 的 L1 损失，支持 NumPy、Torch 和 Pandas 数据类型。

    参数:
    ----------
    treatment : numpy.ndarray, pd.Series, 或 torch.Tensor
        处理组指示变量数组，仅包含 0 和 1。
    y_true : numpy.ndarray, pd.Series, 或 torch.Tensor
        真实结果变量数组，与 treatment 的长度相同。
    y_pred_0 : numpy.ndarray, pd.Series, 或 torch.Tensor
        预测结果变量数组，表示在 treatment = 0 时的潜在结果。
    y_pred_1 : numpy.ndarray, pd.Series, 或 torch.Tensor
        预测结果变量数组，表示在 treatment = 1 时的潜在结果。
    eval_strategy : str
        'observed_only'：基于观测到的分组条件期望计算 ATE_pred。
        'mean_ITE'：基于全体样本的平均 ITE 计算 ATE_pred。

    返回:
    ----------
    float
        ATE 的 L1 损失值。
    """
    # 将输入转换为 NumPy 数组
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()  # 转换 GPU 张量为 NumPy
        return np.asarray(x)

    treatment = to_numpy(treatment)
    y_true = to_numpy(y_true)
    y_pred_0 = to_numpy(y_pred_0)
    y_pred_1 = to_numpy(y_pred_1)

    # 检查输入长度是否一致
    if len(treatment) != len(y_true) or len(y_true) != len(y_pred_0) or len(y_pred_0) != len(y_pred_1):
        raise ValueError("treatment, y_true, y_pred_0 和 y_pred_1 的长度必须一致")

    # 计算真实 ATE（基于真实观测值）
    mask_t1 = treatment == 1
    mask_t0 = treatment == 0
    ATE_true = y_true[mask_t1].mean() - y_true[mask_t0].mean()

    # 根据 eval_strategy 计算预测的 ATE
    if eval_strategy == 'observed_only':
        # 基于观测分组的条件期望
        ATE_pred = y_pred_1[mask_t1].mean() - y_pred_0[mask_t0].mean()
    elif eval_strategy == 'mean_ITE':
        # 基于所有样本的平均 ITE
        ATE_pred = y_pred_1.mean() - y_pred_0.mean()
    else:
        raise ValueError("eval_strategy 必须是 'observed_only' 或 'mean_ITE'")

    # 计算 L1 损失
    ATE_l1_loss = abs(ATE_true - ATE_pred)

    return ATE_l1_loss