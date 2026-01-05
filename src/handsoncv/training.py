import numpy as np
import os
import torch
import time
import wandb
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
from handsoncv.utils import sample_w, sample_flowers
from handsoncv.visualization import log_similarity_heatmap

"""
The following functions are based on the utils provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def count_parameters(model):
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def preprocess_model_input(rgb, lidar, task_mode="fusion", cilp_extras=None):
    """
    Adjusts inputs based on the task.
    Args:
        rgb (torch.Tensor): Image tensor (B, 3, H, W)
        lidar (torch.Tensor): LiDAR tensor (B, C, H, W)
        task_mode (str): Task identifier.
        cilp_extras (dict): Contains pretrained encoders if needed.
    """
    if task_mode in ["fusion", "contrastive"]:
        return rgb, lidar
    elif task_mode == "lidar-only":
        return lidar
    elif task_mode == "fine-tuning":
        return rgb
    elif task_mode == "projector" and cilp_extras is not None:
        for param in cilp_extras['img_enc'].parameters():
            param.requires_grad = False
        cilp_extras['img_enc'].eval() 
        img_emb = cilp_extras['img_enc'](rgb)
        return img_emb
    raise ValueError(f"Unknown task_mode: {task_mode}")

def contrastive_loss(logits, criterion, device):
    """Calculates symmetric Cross Entropy loss for CILP alignment."""
    logits_per_image = logits
    logits_per_lidar = logits.t()
    targets = torch.arange(logits_per_image.size(0), device=device) #Dimension Equal to Batch Size 
    return (criterion(logits_per_image, targets) + criterion(logits_per_lidar, targets)) / 2 # 2 times C.E.

def _train(model, loader, optimizer, criterion, device, task_mode, cilp_extras):
    """Internal helper: Runs one training epoch over the data loader."""
    model.train()
    total_loss = 0
    for step, (rgb, lidar, labels) in enumerate(loader):
        rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
        optimizer.zero_grad()
        
        inputs = preprocess_model_input(rgb, lidar, task_mode, cilp_extras)
        outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        
        if task_mode == "contrastive":
            loss = contrastive_loss(outputs, criterion, device)
        elif task_mode == "projector" and cilp_extras is not None:
            lidar_enc = cilp_extras['lidar_cnn']
            lidar_enc.eval()
            with torch.no_grad():
                target = lidar_enc(lidar, return_embs=True)
            loss = criterion(outputs, target)
        else:
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (step + 1)

def _validate(model, loader, criterion, device, task_mode, cilp_extras, epoch):
    """Internal helper: Runs evaluation and prepares W&B sample table."""
    model.eval()
    val_loss, correct, total = 0, 0, 0
    table = wandb.Table(columns=["Epoch", "Image", "Lidar_Mask", "True_Label", "Predicted_Label"])
    label_names = {0: "cube", 1: "sphere"}
    last_outputs = None

    with torch.no_grad():
        for step, (rgb, lidar, labels) in enumerate(loader):
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
            inputs = preprocess_model_input(rgb, lidar, task_mode, cilp_extras)
            outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            last_outputs = outputs # for Contrastive Visualization

            if task_mode == "contrastive":
                val_loss += contrastive_loss(outputs, criterion, device).item()
            elif task_mode == "projector" and cilp_extras is not None:
                target = cilp_extras['lidar_cnn'](lidar, return_embs=True)
                val_loss += criterion(outputs, target).item()
            else:
                labels = labels.float().unsqueeze(1)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            if step == 0: # Log first 5 prediction samples as wandb table
                for j in range(min(len(labels), 5)):
                    img_vis = wandb.Image(rgb[j][:3].cpu().permute(1,2,0).numpy())
                    lidar_vis = wandb.Image(lidar[j][3].cpu().numpy())
                    p_label = label_names.get(torch.max(outputs, 1)[1][j].item(), "N/A") if total > 0 else "N/A"
                    table.add_data(epoch, img_vis, lidar_vis, label_names[int(labels[j].item())], p_label)

    return val_loss / (step + 1), (100 * correct / max(total, 1)), table, last_outputs

def train_fusion_cilp_model(model, train_loader, val_loader, optimizer, criterion, device, 
                            task_mode="fusion", epochs=10, scheduler=None, cilp_extras=None):
    """
    Universal loop for Task 3, 4, and 5.
    
    Args:
        model (nn.Module): Trained classifier/`Embedder`/crros-modal projector for noise prediction.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        criterion (callable): Loss function used to compute the training and the validation loss.
        optimizer (torch.optim.Optimizer): Optimizer for model updates.
        device (str or torch.device): Device for training.
        task_mode (str, optional): Task identifier (see options below).
        epochs (int, optional): No. of training epochs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        cilp_extras (dict, optional): Contains pretrained encoders if needed.
        
    Returns:
        Dictionary containing metrics, loss and training variables 
    
    task_mode options: 
      - 'fusion': RGB+Lidar classification for different fusion architectures/strategies (Task 3 & Task4)
      - 'lidar-only': Classification using only Lidar images (Task 5.1)
      - 'contrastive': CILP (RGB-LiDAR embeddings Pairs) Alignment (Task 5.1)
      - 'projector': MSE training for the MLP that projects the RGB Embedder output onto the dimensions of the LiDAR Embedder output (Task 5.2)
      - 'fine-tuning': RGB-to-Lidar classification fine-tuning (Task 5.3)
    """
    check_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..", "..")), "Assignment-2", "checkpoints")
    os.makedirs(check_dir, exist_ok=True)

    model.to(device)
    best_val_loss = float('inf') # for Checkpointing Criterion based on Val Loss
    
    # Metrics for the Comparison Table and Static Logs for Wandb
    params = count_parameters(model)
    wandb.config.update({"number_of_parameters": params})
    
    start_time = time.time()
    epoch_times = []
    
    # Reset GPU stats before training starts for new architecture
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)
    
    strategy = wandb.config.get('fusion_strategy', 'default')
    ds_mode = wandb.config.get('downsample_mode', 'default')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training Phase
        avg_train_loss = _train(model, train_loader, optimizer, criterion, device, task_mode, cilp_extras)
        # Validation Phase 
        avg_val_loss, acc, prediction_table, outputs = _validate(model, val_loader, criterion, device, task_mode, cilp_extras, epoch)
        
        # Checkpoint logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_name = f"{task_mode}_{strategy}_{ds_mode}" if task_mode == "fusion" else task_mode
            global_best_path = os.path.join(check_dir, f"{checkpoint_name}_best_model.pt") 
            torch.save(model.state_dict(), global_best_path)
            print(f"Saved new best model to {global_best_path}")
        
        duration = time.time() - epoch_start
        epoch_times.append(duration)
        if torch.cuda.is_available() and "cuda" in str(device):
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # Get GPU Oeak Memory seen since Start of Training Loop
        else:
            peak_mem_mb = 0.0
        
        # Logging Dictionary Update
        log_dict = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time_sec": duration,
            "peak_gpu_mem_mb": peak_mem_mb,
            "sample_predictions": prediction_table
        }
        
        if task_mode == "projector":            
            print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.6f}, Acc: {acc:.2f}% | Mem: {peak_mem_mb:.1f}MB")
        else:
            print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, Acc: {acc:.2f}% | Mem: {peak_mem_mb:.1f}MB")
        if task_mode == "contrastive":
            print("Accuracy not applicable to the embedding alignment task -> Similarity matrix (first 8x8):")
            
            with torch.no_grad():
                cos_sim, cos_sim_wandb = outputs[:8, :8]/model.logit_scale.exp(), outputs[:16, :16]/ model.logit_scale.exp()
                cos_sim, cos_sim_wandb = cos_sim.clamp(-1, 1), cos_sim_wandb.clamp(-1, 1)
                vis, vis_wandb = (cos_sim + 1) / 2, (cos_sim_wandb + 1) / 2 # similarity normalization
                np.set_printoptions(precision=3, suppress=True)
                print(vis.detach().cpu().numpy())
                
                # Diagonal and off-diagonal tracking 
                diag = torch.diag(cos_sim).mean().item()
                off_diag = (cos_sim.sum() - torch.diag(cos_sim).sum()) / (cos_sim.numel() - len(cos_sim))
                print(f"Mean diag: {diag:.3f}, Mean off-diag: {off_diag:.3f}")
                
                fig = log_similarity_heatmap(vis_wandb)
                log_dict["similarity_matrix"] = wandb.Image(fig)
        elif task_mode == "projector":
            print("Accuracy not applicable to the cross-modal projector task")
        
        wandb.log(log_dict)
        
    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    final_peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Final Artifact Logging
    if global_best_path and os.path.exists(global_best_path):
        # Create a unique name for the artifact
        model_artifact = wandb.Artifact(
            name=f"{checkpoint_name}_{wandb.run.id}", 
            type='model',
            description=f"""### Best model checkpoint
            * task: {task_mode}
            * strategy: {strategy}
            * downsampling:** {ds_mode}""",
            metadata={
                "task_mode": task_mode,
                "strategy": strategy,
                "downsample_mode": ds_mode,
                "val_loss": best_val_loss,
                "params": params
            }
        )
        # Add file to artifact and log it to w&b
        model_artifact.add_file(global_best_path) 
        wandb.log_artifact(model_artifact)
        print(f"Successfully logged model artifact: {checkpoint_name}")

    return {
        "val_loss": avg_val_loss,
        "accuracy": acc,
        "params": params,
        "total_time_sec": total_time,
        "sec_per_epoch": avg_epoch_time,
        "gpu_mem_mb": final_peak_gpu
    }

def get_context_mask(c, drop_prob, device):
    """
    Generate a context mask for classifier-free guidance. 
    Each sample has a probability `drop_prob` to have its conditioning 
    dropped (mask = 0) during training, for classifier-free guidance.
    """
    return torch.bernoulli(torch.ones(c.shape[0], 1).to(device) - drop_prob)

def train_diffusion(model, ddpm, train_loader, val_loader, optimizer, epochs, device, 
                    drop_prob, save_dir, sample_save_dir, clip_model, text_list=None):
    """
    Train a DDPM model using classifier-free guidance and save best checkpoints.

    Args:
        model (nn.Module): UNet noise prediction model.
        ddpm (DDPM): Diffusion utility for forward/reverse processes and loss.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer for model updates.
        epochs (int): No. of training epochs.
        device (str or torch.device): Device for training.
        drop_prob (float): Probability to drop conditioning during training.
        save_dir (str): Directory to save best model checkpoint.
        sample_save_dir (str): Directory to save sampled images during training.
        clip_model (CLIP model): Used to encode text prompts for sampling.
        text_list (list of str, optional): Prompts for generating sample images 
            at intervals.
    """
    best_val_loss = float('inf')
    os.makedirs(sample_save_dir, exist_ok=True)
    
    # Helper for sampling during training
    def sample_and_save(epoch_idx):
        if text_list is None: return
        model.eval()
        with torch.no_grad():
            # Call centralized function to generate images via ddpm without embedding stroage
            x_gen, _ = sample_flowers(model, ddpm, clip_model, text_list, device=device, results_dir=sample_save_dir)
            
            grid = make_grid(x_gen.cpu(), nrow=len(text_list), normalize=True, value_range=(-1, 1))
            save_path = os.path.join(sample_save_dir, f"sample_ep{epoch_idx:02d}.png")
            save_image(grid, save_path)
            print(f"Saved samples to {save_path}")
        model.train()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, (x, c) in enumerate(train_loader):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            
            t = torch.randint(0, ddpm.T, (x.shape[0],), device=device).long()
            c_mask = get_context_mask(c, drop_prob, device)
            
            loss = ddpm.get_loss(model, x, t, c, c_mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / (step+1)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (x_v, c_v) in enumerate(val_loader):
                x_v, c_v = x_v.to(device), c_v.to(device)
                t_v = torch.randint(0, ddpm.T, (x_v.shape[0],), device=device).long()
                c_mask_v = get_context_mask(c_v, 0, device) # No dropout in val
                loss_v = ddpm.get_loss(model, x_v, t_v, c_v, c_mask_v)
                val_loss += loss_v.item()
        avg_val = val_loss / (step+1)
        print(f"Epoch {epoch}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # Periodic Sampling
        if epoch % 5 == 0 or epoch == epochs - 1:
            sample_and_save(epoch)

        # Save Best Checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "ddpm_unet_best_model.pt"))
            print(f"--- Saved new best model to {save_dir} ---")