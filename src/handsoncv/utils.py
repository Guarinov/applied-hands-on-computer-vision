import random
import numpy as np
import torch
import os 

def set_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    Parameters
    ----------
    seed : int, optional
        The seed value to use for all random number generators.
        Defaults to 42.
    """
    random.seed(seed) # Python & Random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed) # NumPy seed
    
    torch.manual_seed(seed)  # PyTorch CPU & GPU seeds
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # CuDNN Deterministic algorithms
    # Note: benchmark=False makes it slower but deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed} for reproducibility.")

def seed_worker(worker_id):
    """
    Initialize random seeds for PyTorch DataLoader worker processes.
    It is intended to be passed to the `worker_init_fn` argument of a DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)