import pytest
import torch
import torch.nn.functional as F
import os
import shutil
from unittest.mock import MagicMock, patch
from handsoncv.utils import (
    DDPM, sample_w, sample_flowers, sample_mnist, 
    sample_unconditional, predict_with_cascaded_idk
)

# --- Mocking Helper Classes ---

class MockUNet(torch.nn.Module):
    def __init__(self, img_ch=3, img_size=32, c_embed_dim=512):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.c_embed_dim = c_embed_dim
        self.linear = torch.nn.Linear(1, 1) # Just to have a parameter

    def forward(self, x, t, c=None, c_mask=None):
        # Return same shape as x
        return x * 0.1 

# --- DDPM Tests ---

class TestDDPM:
    @pytest.fixture
    def ddpm_instance(self):
        T = 10
        betas = torch.linspace(1e-4, 0.02, T)
        return DDPM(betas, device=torch.device("cpu"))

    def test_initialization(self, ddpm_instance):
        assert ddpm_instance.T == 10
        assert ddpm_instance.a_bar.shape == (10,)
        assert ddpm_instance.sqrt_a_bar[0] > ddpm_instance.sqrt_a_bar[-1]

    def test_coeff_broadcasting(self, ddpm_instance):
        """Test the helper that reshapes coefficients for (B, C, H, W)."""
        t = torch.tensor([1, 2, 3])
        x_shape = (3, 3, 32, 32)
        coeff = ddpm_instance._get_coeff(ddpm_instance.B, t, x_shape)
        assert coeff.shape == (3, 1, 1, 1)

    def test_forward_diffusion_q(self, ddpm_instance):
        x_0 = torch.randn(2, 3, 32, 32)
        t = torch.tensor([5, 5])
        x_t, noise = ddpm_instance.q(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
        # At t=5, x_t should be different from x_0
        assert not torch.allclose(x_t, x_0)

    def test_reverse_diffusion_step(self, ddpm_instance):
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([5, 5])
        e_t = torch.randn(2, 3, 32, 32)
        
        x_prev = ddpm_instance.reverse_q(x_t, t, e_t)
        assert x_prev.shape == x_t.shape

# --- Sampling Tests ---

class TestSampling:
    @pytest.fixture
    def setup_models(self):
        device = torch.device("cpu")
        betas = torch.linspace(1e-4, 0.02, 5) # Short T for speed
        ddpm = DDPM(betas, device=device)
        model = MockUNet()
        return model, ddpm, device

    def test_sample_w_logic(self, setup_models):
        """Verify CFG sampling shape and batch doubling."""
        model, ddpm, device = setup_models
        c = torch.randn(2, 512) # 2 conditions
        w_tests = [0.0, 1.0]    # 2 weights
        # Expected: 2 conditions * 2 weights = 4 samples
        
        x_gen, x_store = sample_w(
            model, ddpm, (3, 32, 32), ddpm.T, c, device, w_tests=w_tests, store_freq=1
        )
        
        assert x_gen.shape == (4, 3, 32, 32)
        assert len(x_store) > 0

    @patch("clip.tokenize")
    def test_sample_flowers_integration(self, mock_tokenize, setup_models, tmp_path):
        model, ddpm, device = setup_models
        mock_clip = MagicMock()
        mock_clip.encode_text.return_value = torch.randn(2, 512)
        
        results_dir = tmp_path / "flower_results"
        
        x_gen, _ = sample_flowers(
            model, ddpm, mock_clip, ["rose", "lily"], 
            device=device, results_dir=str(results_dir), w_tests=[1.0]
        )
        
        assert x_gen.shape == (2, 3, 32, 32)
        assert os.path.exists(results_dir / "gen_000.png")

    def test_sample_unconditional(self, setup_models):
        model, ddpm, device = setup_models
        # Update model mock to expect 1 channel for MNIST-like test
        model.img_ch = 1
        model.c_embed_dim = 10
        
        x_t = sample_unconditional(model, ddpm, n_samples=4, img_ch=1, img_size=32, device=device)
        assert x_t.shape == (4, 1, 32, 32)

# --- Classifier Utility Tests ---

class TestClassifierUtils:
    def test_cascaded_idk_logic(self):
        """Test both Hard and Soft rejection paths."""
        device = torch.device("cpu")
        
        # Mock model that returns specific logits
        model = MagicMock()
        # Create dummy logits for 3 samples:
        # 1. High confidence digit 0
        # 2. Hard IDK (index 10)
        # 3. Soft IDK (digit 5 but low confidence)
        logits = torch.tensor([
            [10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Class 0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0],  # Class 10 (Hard IDK)
            [0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0],   # Class 5 (Low conf)
        ])
        model.return_value = logits
        
        # Mock loader returning 1 batch
        loader = [(torch.randn(3, 1, 28, 28), torch.tensor([0, 1, 5]))]
        
        results = predict_with_cascaded_idk(
            model, loader, device, threshold=0.9, idk_index=10
        )
        
        assert len(results) == 3
        # Sample 0: Clear digit
        assert results[0]["final_prediction"] == 0
        assert results[0]["rejection_type"] == "None"
        
        # Sample 1: Explicit IDK class
        assert results[1]["final_prediction"] == "IDK"
        assert results[1]["rejection_type"] == "Hard"
        
        # Sample 2: Low confidence digit
        assert results[2]["final_prediction"] == "IDK"
        assert results[2]["rejection_type"] == "Soft"