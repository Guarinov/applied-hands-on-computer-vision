import pytest
import torch
import torch.nn as nn
from handsoncv.models import (
    Embedder, EmbedderStrided, LateFusionNet, IntermediateFusionNet, EfficientCILPModel, CrossModalProjector, LidarClassifier, RGB2LiDARClassifier
)

class TestModels:
    @pytest.fixture
    def dummy_input(self):
        """Provides a standard batch of dummy data."""
        batch_size = 8
        # Dummy RGB input dimension [B, C, H, W]
        rgb = torch.randn(batch_size, 4, 64, 64) 
        # Dummy LiDAR input size [B, C, H, W]
        lidar = torch.randn(batch_size, 4, 64, 64) 
        return rgb, lidar, batch_size

    def test_embedder_variants(self, dummy_input):
        rgb, _, B = dummy_input
        
        # Test Intermediate output (Flattened feature map)
        model_int = Embedder(in_ch=4, return_vector=False)
        out_int = model_int(rgb)
        assert out_int.shape == (B, 200 * 4 * 4)
        
        # Test Intermediate output with Strided Variant as Downsampling Strategy (Flattened feature map)
        model_int = EmbedderStrided(in_ch=4, return_vector=False)
        out_int = model_int(rgb)
        assert out_int.shape == (B, 200 * 4 * 4)
        
        # Test Late output (Bottleneck vector)
        model_late = Embedder(in_ch=4, return_vector=True, emb_dim_late=128)
        out_late = model_late(rgb)
        assert out_late.shape == (B, 128)

    def test_late_fusion_net(self, dummy_input):
        rgb, lidar, B = dummy_input
        model = LateFusionNet(num_classes=2)
        output = model(rgb, lidar)
        assert output.shape == (B, 2)

    @pytest.mark.parametrize("mode", ["concat", "add", "mul"])
    def test_intermediate_fusion_modes(self, dummy_input, mode):
        rgb, lidar, B = dummy_input
        model = IntermediateFusionNet(mode=mode, num_classes=2)
        output = model(rgb, lidar)
        assert output.shape == (B, 2)

    def test_contrastive_models(self, dummy_input):
        rgb, lidar, B = dummy_input
        model = EfficientCILPModel(emb_dim_late=128)
        logits = model(rgb, lidar)
        # Contrastive models output a Similarity Matrix
        assert logits.shape == (B, B)

    def test_modular_rgb2lidar_pipeline(self, dummy_input):
        rgb, _, B = dummy_input
        
        # Create sub-modules
        rgb_enc = Embedder(4, return_vector=True, emb_dim_late=200)
        proj = CrossModalProjector(rgb_dim=200, lidar_dim=3200)
        head = LidarClassifier(emb_dim_interm=200, num_classes=2)
        
        # Assemble
        full_model = RGB2LiDARClassifier(rgb_enc, proj, head)
        output = full_model(rgb)
        assert output.shape == (B, 2)

    def test_backprop_smoke(self, dummy_input):
        """Checks if gradients are computed (no broken links in the graph)."""
        rgb, lidar, B = dummy_input
        model = LateFusionNet(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        output = model(rgb, lidar)
        loss = output.sum() # Dummy loss
        loss.backward()
        
        # Verify that weights in the first layer of the RGB encoder have gradients
        for param in model.rgb_encoder.features.parameters():
            assert param.grad is not None
            break