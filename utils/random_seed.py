import os
import random
import numpy as np
import torch
# import tensorflow


def seeding(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # tensorflow.random.set_seed(seed)
