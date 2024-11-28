# stdlib
import random

# third party
import numpy as np
import torch


def enable_reproducible_results() -> None:
    np.random.seed(19960816)
    torch.manual_seed(19960816)
    random.seed(19960816)
