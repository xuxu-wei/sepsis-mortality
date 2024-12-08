import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10
