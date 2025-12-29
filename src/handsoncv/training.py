import numpy as np
import os
import torch
import time
import wandb
import torch.nn.functional as F

from handsoncv.visualization import log_similarity_heatmap

"""
The following functions are based on the utils provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def search_checkpoint_model(checkpoints_dir, instantiated_model, task_mode='lidar-only'):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def preprocess_model_input(rgb, lidar, task_mode="fusion", cilp_extras=None):
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
    logits_per_image = logits
    logits_per_lidar = logits.t()
    targets = torch.arange(logits_per_image.size(0), device=device) #batch size 
    return (criterion(logits_per_image, targets) + criterion(logits_per_lidar, targets)) / 2 # 2 times C.E.

def train_fusion_cilp_model(model, train_loader, val_loader, optimizer, criterion, device, 
                            task_mode="fusion", epochs=10, scheduler=None, cilp_extras=None):
    """
    Universal loop for Task 3, 4, and 5.
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
    best_val_loss = float('inf') # for checkpoint criterion based on val loss
    
    # Metrics for the comparison table and static logs for wandb
    params = count_parameters(model)
    wandb.config.update({"number_of_parameters": params})
    
    start_time = time.time()
    epoch_times = []
    
    # Reset GPU stats before training starts for new architecture
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training Phase 
        model.train()
        train_loss = 0
        for step, (rgb, lidar, labels) in enumerate(train_loader):
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
            optimizer.zero_grad()
            
            model_inputs = preprocess_model_input(rgb, lidar, task_mode, cilp_extras)
            if isinstance(model_inputs, tuple):
                outputs = model(*model_inputs)
            else:
                outputs = model(model_inputs)
                    
            if task_mode == "contrastive":
                loss = contrastive_loss(outputs, criterion, device)
            elif task_mode == "projector" and cilp_extras is not None:
                for param in cilp_extras['lidar_cnn'].parameters():
                    param.requires_grad = False
                cilp_extras['lidar_cnn'].eval()
                target_lidar_emb = cilp_extras['lidar_cnn'](lidar, return_embs=True) #.flatten(1)
                # target_lidar_emb = F.normalize(target_lidar_emb, dim=1)
                # preds = F.normalize(outputs, dim=1)
                loss = criterion(outputs, target_lidar_emb) #MSE
            else:
                labels = labels.float().unsqueeze(1) 
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging
        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()
        
        avg_train_loss = train_loss / (step+1) #len(train_loader)
            
        # Validation Phase 
        model.eval()
        val_loss, correct, total = 0, 0, 0
        prediction_table = wandb.Table(columns=["Epoch", "Image", "Lidar_Mask", "True_Label", "Predicted_Label"])
        
        label_names = {0: "cube", 1: "sphere"}
                                       
        with torch.no_grad():
            for step, (rgb, lidar, labels) in enumerate(val_loader):
                rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
                                
                model_inputs = preprocess_model_input(rgb, lidar, task_mode, cilp_extras)
                if isinstance(model_inputs, tuple):
                    outputs = model(*model_inputs)
                else:
                    outputs = model(model_inputs)
                    
                if task_mode == "contrastive":
                    val_loss += contrastive_loss(outputs, criterion, device).item() 
                elif task_mode == "projector" and cilp_extras is not None:
                    target_lidar_emb = cilp_extras['lidar_cnn'](lidar, return_embs=True) #.flatten(1)
                    # target_lidar_emb = F.normalize(target_lidar_emb, dim=1)
                    # preds = F.normalize(outputs, dim=1)
                    val_loss += criterion(outputs, target_lidar_emb).item()  #MSE
                else:
                    labels = labels.float().unsqueeze(1) 
                    val_loss += criterion(outputs, labels).item() 
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (predicted == labels).sum().item()
                    # _, predicted = torch.max(outputs.data, 1)
                    # correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    
                # Log sample predictions to W&B (up to 5 samples per epoch, Task 1.3 requirement) 
                if step == 0: # Only from the first batch of val to keep it consistent
                    for j in range(min(len(labels), 5)):
                        img_vis = rgb[j][:3].cpu().permute(1, 2, 0).numpy()
                        lidar_vis = lidar[j][3].cpu().numpy() # Z channel
                        # lidar_vis = lidar[j][0].cpu().numpy() 
                        
                        true_label = label_names[labels[j].item()]
                        if task_mode == "contrastive":
                            p_label = "Aligned"  # contrastive doesn't have class preds
                        elif task_mode == "projector":
                            p_label = "N/A"
                        else: 
                            # (fusion, lidar-only, fine-tuning)
                            pred_idx = torch.max(outputs, 1)[1][j].item()
                            # pred_idx = ((outputs >= 0).long())[j].item()
                            p_label = label_names.get(pred_idx, "unknown")
                        
                        prediction_table.add_data(epoch, wandb.Image(img_vis), wandb.Image(lidar_vis), 
                                                  true_label, p_label)
        
        avg_val_loss = val_loss / (step+1)
        acc = (100 * correct / total) if total > 0 else 0.0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if task_mode != "fusion":
                checkpoint_name = task_mode 
            else: 
                checkpoint_name = f"{task_mode}_{wandb.config['fusion_strategy']}_{wandb.config['downsample_mode']}"
            checkpoint_path = os.path.join(
                check_dir, f"{checkpoint_name}_best_model.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")
        
        duration = time.time() - epoch_start
        epoch_times.append(duration)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # Get peak memory seen since the start of this model's training
        
        log_dict = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "learning_rate": current_lr,
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
                # cos_sim = outputs[:8, :8] 
                cos_sim = outputs[:8, :8]/model.logit_scale.exp()
                # cos_sim_wandb = outputs[:16, :16] 
                cos_sim_wandb = outputs[:16, :16]/ model.logit_scale.exp()
                cos_sim, cos_sim_wandb = cos_sim.clamp(-1, 1), cos_sim_wandb.clamp(-1, 1)
                vis, vis_wandb = (cos_sim + 1) / 2, (cos_sim_wandb + 1) / 2 #cos_sim, cos_sim_wandb #(cos_sim + 1) / 2, (cos_sim_wandb + 1) / 2
                np.set_printoptions(precision=3, suppress=True)
                print(vis.detach().cpu().numpy())
                
                diag = torch.diag(cos_sim).mean().item()
                off_diag = (cos_sim.sum() - torch.diag(cos_sim).sum()) / (cos_sim.numel() - len(cos_sim))
                print(f"Mean diag: {diag:.3f}, Mean off-diag: {off_diag:.3f}")
                
                fig = log_similarity_heatmap(vis_wandb)
                log_dict["similarity_matrix"] = wandb.Image(fig) #wandb.Image(outputs[:16, :16].cpu().detach().numpy())
        elif task_mode == "projector":
            print("Accuracy not applicable to the cross-modal projector task")
        
        wandb.log(log_dict)
        
    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    final_peak_gpu = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "val_loss": avg_val_loss,
        "accuracy": acc,
        "params": params,
        "total_time_sec": total_time,
        "sec_per_epoch": avg_epoch_time,
        "gpu_mem_mb": final_peak_gpu
    }