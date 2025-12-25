import torch
import time
import wandb

"""
The following functions are based on the utils provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_fusion_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')
    
    # Metrics for the comparison table
    params = count_parameters(model)
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        
        for rgb, lidar, labels in train_loader:
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, lidar)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for rgb, lidar, labels in val_loader:
                rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device)
                outputs = model(rgb, lidar)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct / len(val_loader.dataset)
        epoch_time = time.time() - epoch_start
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        wandb.log({
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "epoch_time": epoch_time,
            "gpu_mem_mb": gpu_mem
        })
        
        print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, Acc: {acc:.2f}%")

    total_time = time.time() - start_time
    return {"val_loss": avg_val_loss, "params": params, "total_time": total_time, "gpu_mem": gpu_mem}