import pytest
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from handsoncv.training import (
    train_fusion_cilp_model, train_diffusion
)

# ---- Helper Classes ----

class DummyUNet(nn.Module):
    def __init__(self, img_ch=1, img_size=28, c_embed_dim=10):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.c_embed_dim = c_embed_dim
        # Minimal layer to allow backprop
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, x, t, c, c_mask):
        return x * self.param # Return something the same size as input

class DummyDDPM:
    def __init__(self, T=100):
        self.T = T
    
    def get_loss(self, model, x, t, loss_type, c, c_mask):
        # Return a scalar tensor that depends on model parameters
        pred = model(x, t, c, c_mask)
        return torch.mean((pred - 0)**2)

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

@patch("wandb.config")
@patch("wandb.run")
@patch("wandb.log")
@patch("wandb.Table")
class TestDiffusionTraining:
    @pytest.fixture
    def setup_params(self, tmp_path):
        """Standard parameters for training tests."""
        # Create dummy directories
        save_dir = tmp_path / "checkpoints"
        sample_dir = tmp_path / "samples"
        save_dir.mkdir()
        sample_dir.mkdir()
        
        return {
            "model": DummyUNet(),
            "ddpm": DummyDDPM(),
            "optimizer": None, # Created in test
            "epochs": 1,
            "device": "cpu",
            "drop_prob": 0.1,
            "save_dir": str(save_dir),
            "sample_save_dir": str(sample_dir),
            "loss_type": "mse"
        }

    @pytest.fixture
    def mock_loaders(self):
        """Mock DataLoaders returning (image, conditioning)."""
        batch_size = 2
        img = torch.randn(batch_size, 1, 28, 28)
        
        # Label mode loader
        labels = torch.randint(0, 10, (batch_size,))
        train_label = [(img, labels)]
        
        # Text mode loader
        embs = torch.randn(batch_size, 10)
        train_text = [(img, embs)]
        
        return train_label, train_text

    @patch("wandb.Image")
    @patch("handsoncv.training.sample_mnist")
    @patch("handsoncv.training.get_context_mask")
    @patch("handsoncv.training.count_parameters", return_value=100)
    def test_train_diffusion_mnist_mode(self, mock_count, mock_mask, mock_samp_m, mock_wb_img, 
                                        mock_table, mock_log, mock_run, mock_config,
                                        setup_params, mock_loaders):
        """Tests the training loop in MNIST/Label mode."""
        p = setup_params
        train_ld, _ = mock_loaders
        p["optimizer"] = torch.optim.Adam(p["model"].parameters(), lr=1e-3)
        p["cond_list"] = [0, 1, 2]
        
        mock_samp_m.return_value = (torch.randn(3, 1, 28, 28), None)
        mock_mask.return_value = torch.ones(2, 1)

        train_diffusion(
            model=p["model"], ddpm=p["ddpm"], train_loader=train_ld, val_loader=train_ld,
            optimizer=p["optimizer"], epochs=p["epochs"], device=p["device"],
            drop_prob=p["drop_prob"], save_dir=p["save_dir"], sample_save_dir=p["sample_save_dir"],
            cond_list=p["cond_list"]
        )

        # Verify WandB usage
        assert mock_config.update.called
        assert mock_log.called
        assert mock_samp_m.called

    @patch("handsoncv.training.sample_flowers")
    @patch("handsoncv.training.calculate_clip_score", return_value=0.8)
    def test_train_diffusion_text_mode(self, mock_clip_score, mock_samp_f, 
                                       mock_table, mock_log, mock_run, mock_config, 
                                       setup_params, mock_loaders):
        """Tests the training loop in CLIP/Text mode."""
        p = setup_params
        _, train_ld = mock_loaders
        p["model"] = DummyUNet(c_embed_dim=10) 
        p["optimizer"] = torch.optim.Adam(p["model"].parameters(), lr=1e-3)
        p["cond_list"] = ["rose", "daisy"]
        
        mock_samp_f.return_value = (torch.randn(16, 1, 28, 28), ["rose", "daisy"])

        train_diffusion(
            model=p["model"], ddpm=p["ddpm"], train_loader=train_ld, val_loader=train_ld,
            optimizer=p["optimizer"], epochs=p["epochs"], device=p["device"],
            drop_prob=p["drop_prob"], save_dir=p["save_dir"], sample_save_dir=p["sample_save_dir"],
            cond_list=p["cond_list"], clip_model=MagicMock(), clip_preprocess=MagicMock()
        )

        assert mock_clip_score.called
        assert mock_samp_f.called

    @patch("handsoncv.training.sample_unconditional")
    def test_unconditional_logic(self, mock_samp_u, mock_table, mock_log, mock_run, mock_config,
                                 setup_params, mock_loaders):
        """Tests that the loop runs without a conditioning list."""
        p = setup_params
        train_ld, _ = mock_loaders
        p["optimizer"] = torch.optim.Adam(p["model"].parameters(), lr=1e-3)
        p["cond_list"] = None 
        
        mock_samp_u.return_value = torch.randn(16, 1, 28, 28)
        
        train_diffusion(
            model=p["model"], ddpm=p["ddpm"], train_loader=train_ld, val_loader=train_ld,
            optimizer=p["optimizer"], epochs=p["epochs"], device=p["device"],
            drop_prob=p["drop_prob"], save_dir=p["save_dir"], sample_save_dir=p["sample_save_dir"],
            cond_list=p["cond_list"]
        )
        
        assert mock_samp_u.called
        assert mock_config.update.called