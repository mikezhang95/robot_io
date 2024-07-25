
import time
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir + '/..')
sys.path.append(current_dir)
sys.path.append(parent_dir)

import numpy as np
import torch
import random



def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def np_to_torch(t, device='cpu'):
    if t is None:
        return None
    else:
        return torch.Tensor(t).to(device)

def torch_to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
