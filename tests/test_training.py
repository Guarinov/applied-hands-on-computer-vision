import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from handsoncv.training import train_fusion_cilp_model

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)

@pytest.fixture
def mock_loaders():
    # Batch size 32, 4 channels, 64x64
    rgb = torch.randn(32, 3, 64, 64)
    lidar = torch.randn(32, 1, 64, 64)
    # Combined into 4 channels for 'lidar-only' testing
    combined = torch.randn(32, 4, 64, 64) 
    labels = torch.randint(0, 2, (32,))
    
    # Simple list as a mock dataloader
    dataset = [(combined, combined, labels)]
    return dataset, dataset

@patch("wandb.log")
@patch("wandb.log_artifact")
@patch("wandb.save")
@patch("wandb.Table")
@patch("wandb.config", {"fusion_strategy": "none", "downsample_mode": "none"})
@patch("wandb.run", MagicMock(id="test_run"))
def test_lidar_only_training_loop(mock_table, mock_save, mock_art, mock_log, mock_loaders):
    train_ld, val_ld = mock_loaders
    device = "cpu"
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    results = train_fusion_cilp_model(
        model=model,
        train_loader=train_ld,
        val_loader=val_ld,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        task_mode="lidar-only",
        epochs=1
    )

    assert "val_loss" in results
    assert results["params"] > 0
    # Ensure wandb was actually called
    assert mock_log.called
    assert mock_art.called