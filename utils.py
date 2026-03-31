import random
import numpy as np
import torch
import hashlib
import os



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_train(model):
    device = get_device()
    model.to(device)
    model.train()

def set_eval(model):
    device = get_device()
    model.to(device)
    model.eval()

def get_hash(conf):
    return hashlib.md5(str(conf).encode('utf-8')).hexdigest()

