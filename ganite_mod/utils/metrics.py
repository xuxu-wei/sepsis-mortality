# stdlib
from typing import Tuple

# third party
import torch
import pandas as pd
import numpy as np
from scipy import stats


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - (hat_y[:, 1] - hat_y[:, 0]) ** 2)))


def sqrt_PEHE_with_diff(y: np.ndarray, hat_y: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome difference.
    """
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


def RPol(t: np.ndarray, y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Policy risk(RPol).
    RPol is the average loss in value when treating according to the policy implied by an ITE estimator.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    Output:

    """
    hat_t = np.sign(hat_y[:, 1] - hat_y[:, 0])
    hat_t = 0.5 * (hat_t + 1)
    new_hat_t = np.abs(1 - hat_t)

    # Intersection
    idx1 = hat_t * t
    idx0 = new_hat_t * (1 - t)

    # risk policy computation
    RPol1 = (np.sum(idx1 * y) / (np.sum(idx1) + 1e-8)) * np.mean(hat_t)
    RPol0 = (np.sum(idx0 * y) / (np.sum(idx0) + 1e-8)) * np.mean(new_hat_t)

    return 1 - (RPol1 + RPol0)


def ATE(y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return np.abs(np.mean(y[:, 1] - y[:, 0]) - np.mean(hat_y[:, 1] - hat_y[:, 0]))


def ATT(t: np.ndarray, y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Average Treatment Effect on the Treated(ATT).
    ATT measures what is the expected causal effect of the treatment for individuals in the treatment group.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    """
    # Original ATT
    ATT_value = np.sum(t * y) / (np.sum(t) + 1e-8) - np.sum((1 - t) * y) / (
        np.sum(1 - t) + 1e-8
    )
    # Estimated ATT
    ATT_estimate = np.sum(t * (hat_y[:, 1] - hat_y[:, 0])) / (np.sum(t) + 1e-8)
    return np.abs(ATT_value - ATT_estimate)


def mean_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Generate the mean and a confindence interval over observed data.
    Args:
        data: observed data
        confidence: confidence level
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h


def RCT_ATE(treatment: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Average Treatment Effect (ATE) in a Randomized Controlled Trial (RCT) setting.

    The ATE measures the difference in the expected outcome between the treatment and control groups.
    This metric is only applicable for models trained and evaluated on RCT data.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment assignment vector of shape (n_samples,).
        Contains 1 for treated individuals and 0 for control individuals.
    y : np.ndarray
        Observed outcome vector of shape (n_samples,).

    Returns
    -------
    float
        The Average Treatment Effect (ATE), defined as:
        `ATE = E[Y | T=1] - E[Y | T=0]`

    Raises
    ------
    ValueError
        If the lengths of `treatment` and `y` do not match.

    Notes
    -----
    This function assumes the dataset originates from a randomized controlled trial (RCT),
    ensuring that the treatment assignment is unconfounded.
    """
    # Validate input lengths
    if len(treatment) != len(y):
        raise ValueError("The lengths of `treatment` and `y` must be the same.")

    # Convert inputs to NumPy arrays
    treatment = np.asarray(treatment)
    y = np.asarray(y)

    # Calculate the mean outcomes for treatment (T=1) and control (T=0) groups
    E_y_1 = y[treatment == 1].mean()
    E_y_0 = y[treatment == 0].mean()

    # Compute and return ATE
    return E_y_1 - E_y_0


def RCT_ATE_l1_loss(
    treatment: np.ndarray,
    y_true: np.ndarray,
    y_pred_0: np.ndarray,
    y_pred_1: np.ndarray,
    eval_strategy: str = "observed_only"
) -> float:
    """
    Calculate the L1 loss for ATE estimation in a Randomized Controlled Trial (RCT) setting.

    The L1 loss quantifies the absolute difference between the true ATE (computed from observed data)
    and the predicted ATE (computed from model predictions). This is applicable for models trained
    and evaluated on RCT data.

    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment assignment vector of shape (n_samples,).
        Contains 1 for treated individuals and 0 for control individuals.
    y_true : np.ndarray
        Observed outcome vector of shape (n_samples,).
    y_pred_0 : np.ndarray
        Predicted potential outcomes under control (T=0), of shape (n_samples,).
    y_pred_1 : np.ndarray
        Predicted potential outcomes under treatment (T=1), of shape (n_samples,).
    eval_strategy : str, optional
        Strategy to calculate the predicted ATE:
        - 'observed_only': Computes ATE_pred using only observed treatment groups.
        - 'mean_ITE': Computes ATE_pred as the mean of the predicted individual treatment effects (default is 'observed_only').

    Returns
    -------
    float
        The L1 loss for ATE estimation:
        `L1_loss = |ATE_true - ATE_pred|`
    
    Raises
    ------
    ValueError
        If the lengths of `treatment`, `y_true`, `y_pred_0`, and `y_pred_1` do not match.
        If `eval_strategy` is not one of {'observed_only', 'mean_ITE'}.

    Notes
    -----
    - This function assumes the dataset originates from a randomized controlled trial (RCT),
      ensuring that the treatment assignment is unconfounded.
    - The true ATE is calculated based on observed data, while the predicted ATE depends on the 
      chosen evaluation strategy.

    Examples
    --------
    >>> treatment = np.array([1, 0, 1, 0])
    >>> y_true = np.array([5.0, 3.0, 6.0, 2.0])
    >>> y_pred_0 = np.array([4.5, 3.1, 5.5, 2.2])
    >>> y_pred_1 = np.array([5.5, 4.0, 6.5, 3.0])
    >>> RCT_ATE_l1_loss(treatment, y_true, y_pred_0, y_pred_1)
    0.15
    """
    # Convert inputs to NumPy arrays
    treatment = np.asarray(treatment)
    y_true = np.asarray(y_true)
    y_pred_0 = np.asarray(y_pred_0)
    y_pred_1 = np.asarray(y_pred_1)

    # Validate input lengths
    if not (len(treatment) == len(y_true) == len(y_pred_0) == len(y_pred_1)):
        raise ValueError("The lengths of `treatment`, `y_true`, `y_pred_0`, and `y_pred_1` must match.")

    # Compute true ATE based on observed outcomes
    mask_t1 = treatment == 1
    mask_t0 = treatment == 0
    ATE_true = y_true[mask_t1].mean() - y_true[mask_t0].mean()

    # Compute predicted ATE based on the selected strategy
    if eval_strategy == "observed_only":
        ATE_pred = y_pred_1[mask_t1].mean() - y_pred_0[mask_t0].mean()
    elif eval_strategy == "mean_ITE":
        ATE_pred = y_pred_1.mean() - y_pred_0.mean()
    else:
        raise ValueError("`eval_strategy` must be one of {'observed_only', 'mean_ITE'}.")

    # Compute and return L1 loss
    return np.abs(ATE_true - ATE_pred)