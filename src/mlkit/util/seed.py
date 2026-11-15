import os
import random

import accelerate
import numpy as np
import torch


def seed_everything(seed: int):
    """fix the seed for all the random number generators

    Args:
        seed: the seed to use for the random number generators
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    accelerate.utils.set_seed(seed)
