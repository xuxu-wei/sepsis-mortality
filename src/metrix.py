import pandas as pd
import numpy as np
from math import ceil
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, roc_curve, auc
from lifelines.utils import concordance_index
from scipy import stats
from tqdm import tqdm

def cal_ci(scores, alpha=0.05, method='percentile'):
    """
    Compute the confidence interval for a given array of scores using either 
    a t-distribution-based method or a percentile-based method.

    Parameters
    ----------
    scores : array_like
        A 1D array of numerical scores (e.g., results from k-fold cross-validation).
    alpha : float, optional
        Significance level for the confidence interval. The default is 0.05, 
        which corresponds to a 95% confidence interval. For the percentile method, 
        this parameter defines the coverage (1 - alpha) of the interval.
    method : {'t', 'percentile', 'pct'}, optional
        Method to compute the confidence interval.
        - 't': Use t-distribution-based confidence interval.
        - 'percentile' or 'pct': Use a percentile-based confidence interval.

    Returns
    -------
    mean_score : float
        The mean of the input scores.
    ci_lower : float
        The lower bound of the (1 - alpha)*100% confidence interval.
    ci_upper : float
        The upper bound of the (1 - alpha)*100% confidence interval.

    Examples
    --------
    >>> scores = [0.85, 0.83, 0.87, 0.81, 0.84, 0.86, 0.80, 0.82, 0.88, 0.85]
    >>> mean, lower, upper = cal_ci(scores, alpha=0.05, method='t')
    >>> print(mean, lower, upper)
    0.841 0.8223915514628892 0.8596084485371107

    >>> mean, lower, upper = cal_ci(scores, alpha=0.95, method='percentile')
    >>> print(mean, lower, upper)
    0.841 0.80225 0.87775
    """
    scores = np.array(scores)
    mean_score = np.mean(scores)

    if method == 't':
        # t-distribution-based CI
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        # alpha is significance level, so CI coverage is (1 - alpha)
        coverage = 1 - alpha
        t_val = stats.t.ppf((1 + coverage) / 2, df=n-1)
        ci_lower = mean_score - t_val * (std_score / np.sqrt(n))
        ci_upper = mean_score + t_val * (std_score / np.sqrt(n))

    elif method in ['percentile', 'pct']:
        # percentile-based CI
        coverage = 1 - alpha
        sorted_scores = np.sort(scores)
        lower_perc = (1 - coverage) / 2 * 100
        upper_perc = (1 + coverage) / 2 * 100
        ci_lower = np.percentile(sorted_scores, lower_perc)
        ci_upper = np.percentile(sorted_scores, upper_perc)
    else:
        raise ValueError("method must be either 't' or 'percentile'")

    return mean_score, ci_lower, ci_upper

def format_ci(mean:float, lower:float, upper:float, PREC:int):
    return f'{mean:.{PREC}f} ({lower:.{PREC}f} - {upper:.{PREC}f})'

def get_time_range(times:pd.Series,lower=1,upper=99.95,step=1):
    '''
    describe
    --------
    从生存时间列，根据设定的百分位获取时间范围（用于评价和模型预测）

    parameter
    -------
    times: pd.Series
        生存时间列，dtype=float
    lower: float
        时间区间的低百分位
    upper: float
        时间区间的高百分位
    step: int or float, default 1
        time_range的步长，默认为1
    
    return
    -------
    predict_time_range 

    '''
    lower_bound, upper_bound = np.percentile(times, [lower, upper])
    predict_time_range = np.arange(ceil(lower_bound), int(upper_bound+1), step)
    return predict_time_range


def check_data_validity(y_true_surv, y_true_event, y_pred):
    """
    检查数据的有效性，确保可以计算C-index。

    Parameters
    ----------
    y_true_surv : array-like
        Array of true survival times.
    y_true_event : array-like
        Array of event occurrences (1 if event occurred, 0 otherwise).
    y_pred : array-like
        Array of predicted risks or hazards.

    Returns
    -------
    bool
        True if the data is valid for C-index calculation, False otherwise.
    """
    if len(y_true_surv) < 2:
        return False  # 数据量太小

    if np.all(y_true_event == 0) or np.all(y_true_event == 1):
        return False  # 全部事件或全部未发生事件

    if len(np.unique(y_true_surv)) == 1:
        return False  # 生存时间缺乏多样性

    if len(np.unique(y_pred)) == 1:
        return False  # 预测值缺乏多样性

    return True

def bootstrap_resampler(*data, n_iterations=1000, show_progress=True, random_seed=19960816):
    iter_wrapper = tqdm(range(n_iterations)) if show_progress else range(n_iterations)
    for i in iter_wrapper:
        data_resampled = resample(*data, random_state=random_seed + i)
        yield data_resampled
        
def bootstrap_cindex(y_true_surv, y_true_event, y_pred, n_iterations=1000, show_progress=True, random_seed=19960816):
    """
    Perform bootstrap analysis to estimate the concordance index over multiple resampled datasets.

    Parameters
    ----------
    y_true_surv : array-like
        Array of true survival times.
    y_true_event : array-like
        Array of event occurrences (1 if event occurred, 0 otherwise).
    y_pred : array-like
        Array of predicted risks or hazards.
    n_iterations : int, optional
        Number of bootstrap samples to draw. Default is 1000.
    show_progress : bool, optional
        Whether to show progress bar during bootstrap. Default is True.

    Returns
    -------
    np.ndarray
        Array of bootstrap concordance indices.
    """
    # 在进行抽样前检查数据的有效性
    if not check_data_validity(y_true_surv, y_true_event, y_pred):
        raise ValueError("Input data is not valid for C-index calculation. Please check the survival times, events, and predictions.")

    iter_wrapper = tqdm(range(n_iterations)) if show_progress else range(n_iterations)

    scores = []
    success_count = 0
    for i in iter_wrapper:
        y_true_surv_resampled, y_true_event_resampled, y_pred_resampled = resample(y_true_surv, y_true_event, y_pred, random_state=random_seed + i)
        
        # 在每次重采样后也检查数据有效性
        if not check_data_validity(y_true_surv_resampled, y_true_event_resampled, y_pred_resampled):
            continue

        try:
            current_score = concordance_index(y_true_surv_resampled, -y_pred_resampled, y_true_event_resampled)
            scores.append(current_score)
            success_count += 1
        except ZeroDivisionError:
            # 跳过无法计算 C-index 的情况
            continue
    
    if len(scores) == 0:
        raise ValueError("No valid C-index could be calculated. Please check your data.")

    if success_count <= 0.3 * n_iterations:
        UserWarning(f"有效重采样次数不足30% ({0.3 * n_iterations:,}), 请注意boostrap估计有效性。")

    return np.array(scores)

def bootstrap_auc(y_true, y_pred, n_iterations=1000, show_progress=True, random_seed=19960816):
    """
    Perform bootstrap analysis to estimate the concordance index over multiple resampled datasets.

    Parameters
    ----------
    y_true_surv : array-like
        Array of true survival times.
    y_true_event : array-like
        Array of event occurrences (1 if event occurred, 0 otherwise).
    y_pred : array-like
        Array of predicted risks or hazards.
    n_iterations : int, optional
        Number of bootstrap samples to draw. Default is 1000.
    show_progress : bool, optional
        Whether to show progress bar during bootstrap. Default is True.

    Returns
    -------
    np.ndarray
        Array of bootstrap concordance indices.
    """
    iter_wrapper = tqdm(range(n_iterations)) if show_progress else range(n_iterations)

    scores = []
    for i in iter_wrapper:
        y_true_resampled, y_pred_resampled = resample(y_true, y_pred, random_state=random_seed + i)
        current_score = roc_auc_score(y_true_resampled, y_pred_resampled)
        scores.append(current_score)
    return np.array(scores)


def bootstrap_cindex_with_report(y_true_surv, y_true_event, y_pred, alpha=0.95, n_iterations=1000, show_progress=True):
    """
    Calculate the bootstrap C-index and confidence interval on the test set.

    Parameters
    ----------
    y_true_surv : array-like
        Array of true survival times.
    y_true_event : array-like
        Array of event occurrences (1 if event occurred, 0 otherwise).
    y_pred : array-like
        Array of predicted risks or hazards.
    alpha : float, optional
        Confidence level. Default is 0.95.
    n_iterations : int, optional
        Number of bootstrap samples to draw. Default is 1000.

    Returns
    -------
    np.ndarray
        Array of bootstrap C-index scores.
    """
    # Calculate bootstrap C-index
    cindex_array = bootstrap_cindex(y_true_surv, y_true_event, np.array(y_pred), n_iterations=n_iterations, show_progress=show_progress, random_seed=19960816)
    cindex_mean, cindex_lower, cindex_upper = cal_ci(cindex_array, alpha=alpha)
    cindex_std = np.std(cindex_array)

    print(f'bootstrap C-index [mean ({alpha*100}% CI)]: {cindex_mean:.4f} ({cindex_lower:.4f} - {cindex_upper:.4f})')
    print(f'bootstrap C-index [mean (SD)]: {cindex_mean:.4f} ({cindex_std:.4f})')
    return cindex_array

