import os
import torch
import random
import numpy as np

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()