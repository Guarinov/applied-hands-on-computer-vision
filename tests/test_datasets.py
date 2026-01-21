import pytest
import torch
import csv
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from handsoncv.datasets import (
    CILPFusionDataset, TFflowersCLIPDataset, GeneratedMNISTDataset
)

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
        
class TestTFflowersCLIPDataset:
    @pytest.fixture
    def mock_clip_data(self, tmp_path):
        """Creates dummy images and a CSV file mapping paths to embeddings."""
        img_dir = tmp_path / "flowers"
        img_dir.mkdir()
        csv_path = tmp_path / "clip.csv"
        
        data_rows = []
        emb_dim = 128
        
        for i in range(5):
            img_path = img_dir / f"flower_{i}.jpg"
            # Create dummy RGB image
            Image.new('RGB', (32, 32), color='red').save(img_path)
            
            # Create dummy embedding [img_path, emb1, emb2, ...]
            embedding = np.random.randn(emb_dim).tolist()
            data_rows.append([str(img_path)] + embedding)
            
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_rows)
            
        return csv_path, emb_dim

    def test_initialization_and_len(self, mock_clip_data):
        csv_path, _ = mock_clip_data
        dataset = TFflowersCLIPDataset(csv_path=str(csv_path), transform=None)
        assert len(dataset) == 5

    def test_getitem_shapes(self, mock_clip_data):
        csv_path, emb_dim = mock_clip_data
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        dataset = TFflowersCLIPDataset(csv_path=str(csv_path), transform=transform)
        
        img, emb = dataset[0]
        
        # Check image: Resize to 224, ToTensor makes it (3, 224, 224)
        assert img.shape == (3, 224, 224)
        # Check embedding: length matches CSV columns
        assert emb.shape == (emb_dim,)
        assert isinstance(emb, torch.Tensor)

class TestGeneratedMNISTDataset:
    @pytest.fixture
    def mock_mnist_results(self, tmp_path):
        """Creates dummy grayscale images and a results list."""
        img_dir = tmp_path / "generated"
        img_dir.mkdir()
        
        results_list = []
        # 3 valid samples
        for i in range(3):
            path = img_dir / f"gen_{i}.png"
            Image.new('L', (28, 28), color=128).save(path)
            results_list.append({'img_path': str(path), 'classifier_label': i})
            
        # 1 'IDK' sample
        path_idk = img_dir / "gen_idk.png"
        Image.new('L', (28, 28), color=255).save(path_idk)
        results_list.append({'img_path': str(path_idk), 'classifier_label': 'IDK'})
        
        # 1 invalid sample (no image path)
        results_list.append({'img_path': None, 'classifier_label': 5})
        
        return results_list

    def test_filtering_logic(self, mock_mnist_results):
        """Ensure samples with img_path=None are ignored."""
        dataset = GeneratedMNISTDataset(results_list=mock_mnist_results)
        # 3 valid + 1 IDK = 4 total
        assert len(dataset) == 4

    def test_getitem_content(self, mock_mnist_results):
        transform = transforms.ToTensor()
        dataset = GeneratedMNISTDataset(results_list=mock_mnist_results, transform=transform)
        
        img, label = dataset[0]
        # Check image shape (Grayscale 'L' -> 1 channel)
        assert img.shape == (1, 28, 28)
        assert isinstance(label, int)
        
        # Check 'IDK' label
        img_idk, label_idk = dataset[3]
        assert label_idk == 'IDK'

    def test_dataloader_integration(self, mock_mnist_results):
        dataset = GeneratedMNISTDataset(results_list=mock_mnist_results, transform=transforms.ToTensor())
        # Batch size 4 should work despite mixed int/string labels if handled 
        loader = DataLoader(dataset, batch_size=1) 
        img, label = next(iter(loader))
        assert img.shape == (1, 1, 28, 28)