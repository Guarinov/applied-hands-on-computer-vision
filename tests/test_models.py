import pytest
import torch
import torch.nn as nn
from handsoncv.models import (
    Embedder, EmbedderStrided, LateFusionNet, IntermediateFusionNet, EfficientCILPModel, 
    CrossModalProjector, LidarClassifier, RGB2LiDARClassifier, UNet, MnistClassifier
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
        
class TestUNet:
    @pytest.fixture
    def unet_params(self):
        return {
            "T": 1000,
            "img_ch": 1,
            "img_size": 32,
            "down_chs": (32, 64, 128),
            "t_embed_dim": 8,
            "c_embed_dim": 128
        }

    def test_unet_forward_shape(self, unet_params):
        """Verify UNet output shape matches input image shape."""
        model = UNet(**unet_params)
        B = 4
        x = torch.randn(B, unet_params["img_ch"], unet_params["img_size"], unet_params["img_size"])
        t = torch.randint(0, unet_params["T"], (B,))
        c = torch.randn(B, unet_params["c_embed_dim"])
        c_mask = torch.ones(B, unet_params["c_embed_dim"])

        output = model(x, t, c, c_mask)
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_unet_self_attention(self, unet_params):
        """Verify UNet works with the self-attention bottleneck enabled."""
        model = UNet(**unet_params, self_attention=True, num_heads=4)
        B = 2
        x = torch.randn(B, unet_params["img_ch"], unet_params["img_size"], unet_params["img_size"])
        t = torch.randint(0, unet_params["T"], (B,))
        c = torch.randn(B, unet_params["c_embed_dim"])
        c_mask = torch.ones(B, unet_params["c_embed_dim"])

        output = model(x, t, c, c_mask)
        assert output.shape == x.shape

    def test_unet_conditioning_mask(self, unet_params):
        """Verify that the conditioning mask zeros out the context."""
        model = UNet(**unet_params)
        B = 2
        x = torch.randn(B, unet_params["img_ch"], unet_params["img_size"], unet_params["img_size"])
        t = torch.randint(0, unet_params["T"], (B,))
        c = torch.randn(B, unet_params["c_embed_dim"])
        
        # Pass all zeros as mask
        c_mask_zero = torch.zeros(B, unet_params["c_embed_dim"])
        out_masked = model(x, t, c, c_mask_zero)
        
        # Pass different context but same zero mask
        c_different = torch.randn(B, unet_params["c_embed_dim"])
        out_different_context = model(x, t, c_different, c_mask_zero)
        
        # Outputs should be identical because context was masked to zero
        assert torch.allclose(out_masked, out_different_context, atol=1e-5)

    def test_unet_backprop(self, unet_params):
        """Smoke test for gradients in the UNet."""
        model = UNet(**unet_params)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 1000, (2,))
        c = torch.randn(2, 128)
        c_mask = torch.ones(2, 128)
        
        out = model(x, t, c, c_mask)
        loss = out.mean()
        loss.backward()
        
        # Check if gradients exist in the first downsampling layer
        for param in model.down0.parameters():
            assert param.grad is not None
            break

class TestMnistClassifier:
    def test_classifier_shape(self):
        """Verify LeNet-5 output classes."""
        num_classes = 10
        model = MnistClassifier(num_classes=num_classes)
        # MNIST images are 1x28x28
        x = torch.randn(8, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (8, num_classes)

    def test_classifier_idk_class(self):
        """Verify classifier works with an extra 'IDK' class."""
        model = MnistClassifier(num_classes=11)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 11)

    def test_classifier_invalid_input(self):
        """LeNet-5 is sensitive to input size due to fixed kernel sizes."""
        model = MnistClassifier(num_classes=10)
        # Passing 32x32 instead of 28x28 should usually trigger a shape error 
        # at the View/Linear layer in this specific architecture
        with pytest.raises(RuntimeError):
            x = torch.randn(1, 1, 32, 32)
            model(x)

    def test_classifier_backprop(self):
        """Check if gradients flow through the classifier."""
        model = MnistClassifier(num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 1, 28, 28)
        labels = torch.randint(0, 10, (4,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        
        # Check first conv layer
        assert model.conv1.weight.grad is not None