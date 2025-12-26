import torch
import time
import wandb

"""
The following functions are based on the utils provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    model.to(device)
    best_val_loss = float('inf')
    
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
        for rgb, lidar, labels in train_loader:
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
            optimizer.zero_grad()

            if task_mode in ["fusion", "contrastive"]:
                outputs = model(rgb, lidar)
            elif task_mode == "lidar-only":
                outputs = model(lidar)
            elif task_mode == "projector":
                # cilp_extras should contain sfrozen embedders
                with torch.no_grad():
                    img_emb = cilp_extras['img_enc'](rgb)
                    target_lidar_emb = cilp_extras['lidar_cnn'].embedder(lidar).flatten(1)
                outputs = model(img_emb)
            elif task_mode == "fine-tuning":
                outputs = model(rgb)
            
            if task_mode == "contrastive":
                # CLIP Loss logic adapted to CILP (RGB-LiDAR) setting
                logits_per_image = outputs
                logits_per_lidar = outputs.t()
                ground_truth = torch.arange(len(rgb), device=device)
                loss = (criterion(logits_per_image, ground_truth) + 
                        criterion(logits_per_lidar, ground_truth)) / 2 # 2 CE
            elif task_mode == "projector":
                loss = criterion(outputs, target_lidar_emb) #MSE
            else:
                loss = criterion(outputs, labels) #CE
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging
        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
            
        # Validation Phase 
        model.eval()
        val_loss, correct, total = 0, 0, 0
        prediction_table = wandb.Table(columns=["Epoch", "Image", "Lidar_Mask", "True_Label", "Predicted_Label"])
        
        label_names = {0: "cube", 1: "sphere"}
                                       
        with torch.no_grad():
            for step, (rgb, lidar, labels) in enumerate(val_loader):
                rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
                
                # Logic same as training for outputs
                if task_mode in ["fusion", "contrastive"]: outputs = model(rgb, lidar)
                elif task_mode == "lidar_only": 
                    outputs = model(lidar) 
                elif task_mode == "projector":
                    img_emb = cilp_extras['img_enc'](rgb)
                    target_lidar_emb = cilp_extras['lidar_cnn'].embedder(lidar).flatten(1)
                    outputs = model(img_emb)
                elif task_mode == "fine-tuning": outputs = model(rgb)
                
                # Loss
                if task_mode == "contrastive":
                    gt = torch.arange(len(rgb), device=device)
                    val_loss += ((criterion(outputs, gt) + criterion(outputs.t(), gt)) / 2).item() #2 CE
                elif task_mode == "projector":
                    val_loss += criterion(outputs, target_lidar_emb).item()
                else:
                    val_loss += criterion(outputs, labels).item() 
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    
                # Log sample predictions to W&B (up to 5 samples per epoch, Task 1.3 requirement) 
                if step == 0: # Only from the first batch of val to keep it consistent
                    for j in range(min(len(labels), 5)):
                        img_vis = rgb[j][:3].cpu().permute(1, 2, 0).numpy()
                        lidar_vis = lidar[j][3].cpu().numpy() # Z channel
                        
                        true_label = label_names[labels[j].item()]
                        if task_mode == "contrastive":
                            p_label = "Aligned"  # contrastive doesn't have class preds
                        elif task_mode == "projector":
                            p_label = "N/A"
                        else: 
                            # (fusion, lidar-only, fine-tuning)
                            pred_idx = torch.max(outputs, 1)[1][j].item()
                            p_label = label_names.get(pred_idx, "unknown")
                        
                        prediction_table.add_data(epoch, wandb.Image(img_vis), wandb.Image(lidar_vis), 
                                                  true_label, p_label)

        avg_val_loss = val_loss / len(val_loader)
        acc = (100 * correct / total) if total > 0 else 0
        
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
        
        if task_mode == "contrastive":
            log_dict["similarity_matrix"] = wandb.Image(outputs[:16, :16].cpu().detach().numpy())
        
        wandb.log(log_dict)
        
        print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, Acc: {acc:.2f}% | Mem: {peak_mem_mb:.1f}MB")
        
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