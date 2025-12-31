import random
import numpy as np
import torch
import os 

def get_binary_correct(output, target, threshold=0.5):
    """
    Computes the number of correct predictions for binary classification.
    
    Args:
        output (torch.Tensor): Raw logits from the model.
        target (torch.Tensor): Ground truth labels.
        threshold (float): Sigmoid threshold for positive class. Default: 0.5.
        
    Returns:
        int: Number of correct predictions in the current batch.
    """
    probs = torch.sigmoid(output)
    preds = (probs > threshold).float()
    return (preds == target).sum().item()

def evaluate_model(model, val_loader, criterion, device, run_final_assessment=False):
    """
    Evaluates the model on the validation set and calculates average loss and accuracy.
    
    Args:
        model (nn.Module): The cross-modal classifier (RGB2LiDARClassifier).
        val_loader (DataLoader): Validation data loader.
        criterion (Callable): The loss function (e.g., BCEWithLogitsLoss).
        device (str/torch.device): Device to run evaluation on.
        run_final_assessment (bool): If True, only evaluate the first 5 batches and check 95% threshold.
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    # total_samples = len(val_loader.dataset)
    total = 0.0
    
    if run_final_assessment:
        print("Running Final Assessment: Evaluating only the first 5 batches...")

    with torch.no_grad():
        for step, (r_img, _, c_idx) in enumerate(val_loader):
            # If final assessment is on, stop after 5 batches (indices 0 to 4)
            if run_final_assessment and step >= 5:
                break
            
            # Assume (rgb mg, lidar img, class index) as batch tuple
            # We also modify the class index format to make it suitable for BCE loss
            r_img, c_idx = r_img.to(device), c_idx.to(device).float().unsqueeze(1)
            
            out = model(r_img)
            loss = criterion(out, c_idx)
            
            running_loss += loss.item() * r_img.size(0)
            total += c_idx.size(0)
            total_correct += get_binary_correct(out, c_idx)
    
    avg_loss = running_loss / total
    accuracy = total_correct / total        
    final_acc_pct = accuracy * 100
    # avg_loss = running_loss / total_samples
    # accuracy = total_correct / total_samples
    
    if run_final_assessment:
        if final_acc_pct > 95.0:
            print(f"✅ Success! Accuracy is {final_acc_pct:.2f}%, which is above 95%")
        else:
            print(f"❌ Accuracy {final_acc_pct:.2f}% is below 95%. Revise the fine-tuning strategy or inspect the reason")
    else:
        print(f"Validation Results - Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    
    # return avg_loss, accuracy

def search_checkpoint_model(checkpoints_dir, instantiated_model, task_mode='lidar-only'):
    """
    Loads a pretrained model checkpoint if available and sets appropriate training modes.

    Parameters
    ----------
    checkpoints_dir : str
        Directory containing saved model checkpoints.
    instantiated_model : nn.Module
        PyTorch model instance to load the checkpoint into.
    task_mode : str, optional
        Task type, which determines behavior:
        - 'lidar-only', 'contrastive', etc.: model set to `.eval()` with frozen parameters.
        - 'projector': model remains in `.train()` mode for fine-tuning.

    Returns
    -------
    nn.Module
        The model loaded with checkpoint weights (if found) and set to the correct mode.
    """
    checkpoint_path = os.path.join(checkpoints_dir, f"{task_mode}_best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        instantiated_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded {task_mode} model from checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Using instantiated model as is.")

    # Freeze parameters and set eval mode for cross-modal projector training
    # if task_mode == 'contrastive':
    if task_mode != 'projector':
        for param in instantiated_model.parameters():
            param.requires_grad = False
        instantiated_model.eval()
        print(f"        Using instantiated model in `.eval()` mode.")
        return instantiated_model
    else:
        instantiated_model.train()
        print(f"        Using instantiated model in `.train()` mode.")
        return instantiated_model

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