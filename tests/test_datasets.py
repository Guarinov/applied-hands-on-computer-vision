import pytest
import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from handsoncv.datasets import CILPFusionDataset

class TestCILPFusionDataset:
    @pytest.fixture(scope="class")
    def mock_root(self, tmp_path_factory):
        """Creates a temporary mock dataset structure for the whole class."""
        tmp_dir = tmp_path_factory.mktemp("data")
        img_size = 64
        
        for folder in ["cubes", "spheres"]:
            base = tmp_dir / folder
            (base / "rgb").mkdir(parents=True)
            (base / "lidar").mkdir(parents=True)
            
            np.save(base / "azimuth.npy", np.random.rand(img_size).astype(np.float32)) # This matches the broadcasting logic [:, None] in get_torch_xyza
            np.save(base / "zenith.npy", np.random.rand(img_size).astype(np.float32))
            
            for i in range(3): 
                stem = f"{i:04d}"
                img_data = np.random.randint(0, 255, (img_size, img_size, 4), dtype=np.uint8)
                Image.fromarray(img_data).save(base / "rgb" / f"{stem}.png")
                np.save(base / "lidar" / f"{stem}.npy", np.random.rand(img_size, img_size).astype(np.float32))
        return tmp_dir

    def test_initialization(self, mock_root):
        dataset = CILPFusionDataset(root_dir=str(mock_root))
        assert len(dataset) == 6

    def test_single_item_shapes(self, mock_root):
        """Verify the 4-channel (4, 64, 64) output for both RGB and LiDAR."""
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CILPFusionDataset(root_dir=str(mock_root), transform=transform)
        
        rgb, lidar, label = dataset[0]
        
        # Check shapes: (C, H, W)
        assert rgb.shape == (4, 64, 64), f"Expected (4, 64, 64), got {rgb.shape}"
        assert lidar.shape == (4, 64, 64), f"Expected (4, 64, 64), got {lidar.shape}"
        assert isinstance(label, torch.Tensor)

    def test_dataloader_batching(self, mock_root):
        """Verify batching: (Batch, 4, 64, 64)."""
        batch_size = 2
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CILPFusionDataset(root_dir=str(mock_root), transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        batch_rgb, batch_lidar, batch_labels = next(iter(loader))
        
        assert batch_rgb.shape == (batch_size, 4, 64, 64)
        assert batch_lidar.shape == (batch_size, 4, 64, 64)
        assert batch_labels.shape == (batch_size,)
    
    def test_label_mapping(self, mock_root):
        """Ensure cube maps to 0 and sphere maps to 1."""
        dataset = CILPFusionDataset(root_dir=str(mock_root))
        # Find a cube sample index
        cube_idx = next(i for i, sid in enumerate(dataset.sample_ids) if "cube" in sid)
        # Find a sphere sample index
        sphere_idx = next(i for i, sid in enumerate(dataset.sample_ids) if "sphere" in sid)
        
        _, _, cube_label = dataset[cube_idx]
        _, _, sphere_label = dataset[sphere_idx]
        
        assert cube_label == 0
        assert sphere_label == 1