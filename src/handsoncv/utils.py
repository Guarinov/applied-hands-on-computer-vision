import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import clip
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
from handsoncv.visualization import show_tensor_image

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

"""
The following UNet components' functions are based on the modules provided for the Nvidia course
Generative AI with Diffusion Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-08+V1 
"""

class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) utility class.

    Implements forward diffusion, reverse diffusion, loss computation,
    and sampling utilities based on NVIDIA's diffusion course.
    """
    def __init__(self, B, device):
        """       
        Input:
            B      : Tensor (T,) – noise schedule (betas)
            device : torch.device
        """
        self.B = B
        self.T = len(B)
        self.device = device

        # Forward diffusion variables
        self.a = 1.0 - self.B
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.sqrt_a_bar = torch.sqrt(self.a_bar)  # Mean Coefficient
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - self.a_bar)  # St. Dev. Coefficient

        # Reverse diffusion variables
        self.sqrt_a_inv = torch.sqrt(1 / self.a)
        self.pred_noise_coeff = (1 - self.a) / torch.sqrt(1 - self.a_bar)

    def q(self, x_0, t):
        """
        Forward diffusion process q(x_t | x_0).

        Input:
            x_0 : Tensor (B, C, H, W) – clean image
            t   : Tensor (B,) – timestep

        Output:
            x_t   : Tensor (B, C, H, W) – noisy image
            noise : Tensor (B, C, H, W) – sampled noise
        """
        t = t.int()
        noise = torch.randn_like(x_0)
        sqrt_a_bar_t = self.sqrt_a_bar[t, None, None, None]
        sqrt_one_minus_a_bar_t = self.sqrt_one_minus_a_bar[t, None, None, None]

        x_t = sqrt_a_bar_t * x_0 + sqrt_one_minus_a_bar_t * noise
        return x_t, noise

    def get_loss(self, model, x_0, t, *model_args):
        """
        Compute DDPM training loss (MSE on predicted noise).

        Input:
            model      : noise prediction model
            x_0        : Tensor (B, C, H, W)
            t          : Tensor (B,)
            model_args : additional conditioning inputs

        Output:
            Tensor – scalar loss
        """
        x_noisy, noise = self.q(x_0, t)
        noise_pred = model(x_noisy, t, *model_args)
        return F.mse_loss(noise, noise_pred)

    @torch.no_grad()
    def reverse_q(self, x_t, t, e_t):
        """
        Reverse diffusion step q(x_{t-1} | x_t).

        Input:
            x_t : Tensor (B, C, H, W) – noisy image at timestep t
            t   : Tensor (B,)
            e_t : Tensor (B, C, H, W) – predicted noise

        Output:
            Tensor (B, C, H, W) – denoised image at timestep t-1
        """
        t = t.int()
        pred_noise_coeff_t = self.pred_noise_coeff[t]
        sqrt_a_inv_t = self.sqrt_a_inv[t]
        u_t = sqrt_a_inv_t * (x_t - pred_noise_coeff_t * e_t)
        if t[0] == 0:  # All t values should be the same
            return u_t  # Reverse diffusion complete!
        else:
            B_t = self.B[t - 1]  # Apply noise from the previos timestep
            new_noise = torch.randn_like(x_t)
            return u_t + torch.sqrt(B_t) * new_noise

    @torch.no_grad()
    def sample_images(self, model, img_ch, img_size, ncols, *model_args, axis_on=False):
        """
        Generate and visualize samples by reverse diffusion.

        Input:
            model      : trained noise prediction model
            img_ch     : number of image channels
            img_size   : spatial resolution (H = W)
            ncols      : number of visualization columns
            model_args : conditioning inputs
            axis_on    : whether to show plot axes
        """
        # Noise to generate images from
        x_t = torch.randn((1, img_ch, img_size, img_size), device=self.device)
        plt.figure(figsize=(8, 8))
        hidden_rows = self.T / ncols
        plot_number = 1

        # Go from T to 0 removing and adding noise until t = 0
        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, device=self.device).float()
            e_t = model(x_t, t, *model_args)  # Predicted noise
            x_t = self.reverse_q(x_t, t, e_t)
            if i % hidden_rows == 0:
                ax = plt.subplot(1, ncols+1, plot_number)
                if not axis_on:
                    ax.axis('off')
                show_tensor_image(x_t.detach().cpu())
                plot_number += 1
        plt.show()

@torch.no_grad()
def sample_w(
    model, ddpm, input_size, T, c, device, w_tests=None, store_freq=10
):
    """
    Classifier-free guidance sampling with varying guidance weights.

    Input:
        model      : trained noise prediction model
        ddpm       : DDPM instance
        input_size : tuple (C, H, W)
        T          : number of diffusion steps
        c          : Tensor (N, C_dim) – conditioning vectors
        device     : torch.device
        w_tests    : list of guidance weights
        store_freq : timestep interval for storing samples

    Output:
        x_t        : Tensor (N, C, H, W) – final samples
        x_t_store  : Tensor (K, N, C, H, W) – intermediate samples
    """
    if w_tests is None:
        w_tests = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    # Preprase "grid of samples" with w for rows and c for columns
    n_samples = len(w_tests) * len(c)

    # One w for each c
    w = torch.tensor(w_tests).float().repeat_interleave(len(c))
    w = w[:, None, None, None].to(device)  # Make w broadcastable
    x_t = torch.randn(n_samples, *input_size).to(device)

    # One c for each w
    c = c.repeat(len(w_tests), 1)

    # Double the batch
    c = c.repeat(2, 1)

    # Don't drop context at test time
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    x_t_store = []
    for i in range(0, T)[::-1]:
        # Duplicate t for each sample
        t = torch.tensor([i]).to(device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch
        x_t = x_t.repeat(2, 1, 1, 1)
        t = t.repeat(2, 1, 1, 1)

        # Find weighted noise
        e_t = model(x_t, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        e_t = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Deduplicate batch for reverse diffusion
        x_t = x_t[:n_samples]
        t = t[:n_samples]
        x_t = ddpm.reverse_q(x_t, t, e_t)

        # Store values for animation
        if i % store_freq == 0 or i == T or i < 10:
            x_t_store.append(x_t)

    x_t_store = torch.stack(x_t_store)
    return x_t, x_t_store

def sample_flowers(unet_model, ddpm, clip_model, text_list, device="cuda" if torch.cuda.is_available() else "cpu", 
                   results_dir=None, embeddings_storage=None, w_tests=None):
    """
    Generate flower images via a trained UNet/DDPM conditioned on CLIP text embeddings.

    Args:
        unet_model (nn.Module): Trained UNet model for noise prediction.
        ddpm (DDPM): DDPM utility with forward/reverse diffusion functions.
        clip_model (CLIP model): CLIP model used to encode text prompts.
        text_list (list of str): Text prompts to condition image generation.
        return_gen_single_img (bool, optional): if True, saves and returns individual 
            image tensors instead of batches.
        device (str or torch.device): Device for computation.
        results_dir (str, optional): Folder to save images if `return_gen_single_img=True`.
        embeddings_storage (object, optional): Optional hook used to collect UNet embeddings. 
            If provided, final forward pass at t=0 is executed to trigger embedding 
            extraction.

    Returns:
        If return_gen_single_img:
            None : either save individual image tensors or store bottlneck embeddings.
        Else:
            x_gen : Tensor (B, C, H, W)L final batch of generated images.
            x_gen_store : Tensor (K, B, C, H, W): stored intermediate images for visualization.
    """
    unet_model.eval()
    text_tokens = clip.tokenize(text_list).to(device)
    
    with torch.no_grad():
        c = clip_model.encode_text(text_tokens).float()
        input_size = (unet_model.img_ch, unet_model.img_size, unet_model.img_size)
        # Sample images using classifier-free guidance
        x_gen, x_gen_store = sample_w(unet_model, ddpm, input_size, ddpm.T, c, device, w_tests)
        
        # Extract Embeddings (If storage provided)
        # We run one final pass on the clean generated images to get thebottleneck
        if embeddings_storage is not None:
            repeat_factor = x_gen.shape[0] // c.shape[0]
            # One c for each x_gen given variable w_tests shape
            c_repeated = c.repeat(repeat_factor, 1) # shape is [21, 512]
            t_zero = torch.zeros(x_gen.shape[0], device=device).long()
            c_mask = torch.ones(x_gen.shape[0], 1, device=device)
            _ = unet_model(x_gen, t_zero, c_repeated, c_mask) 

        # Save images if results_dir is provided
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            for i in range(len(text_list)):
                # Save as image file
                img_path = os.path.join(results_dir, f"gen_{i:03d}.png")
                # Rescale from [-1, 1] to [0, 1] for saving
                save_image(x_gen[i:i+1], img_path, normalize=True, value_range=(-1, 1))  # [1, 3, 32, 32]
    return x_gen, x_gen_store