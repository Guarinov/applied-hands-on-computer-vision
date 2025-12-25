import os 
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

"""
The following functions are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def get_lidar_projections(lidar_depth, azimuth, zenith):
    """Calculate X, Y, Z projections based on Nvidia/Omniverse LiDAR math."""
    # Create the projections using broadcasting
    # Azimuth: rows ([:, None]), Zenith: columns ([None, :])
    x = lidar_depth * np.sin(-azimuth[:, None]) * np.cos(-zenith[None, :])
    y = lidar_depth * np.cos(-azimuth[:, None]) * np.cos(-zenith[None, :])
    z = lidar_depth * np.sin(-zenith[None, :])
    
    # Create mask (a) for valid returns (max range 50.0 for this specific dataset; oterhwsie adjust accordingly)
    # This matches the 'a' in the Nvidia get_torch_xyza function
    mask = (lidar_depth < 50.0).astype(np.float32)
    return x, y, z, mask

def create_lidar_viz(npy_path, azimuth, zenith, output_path):
    """Converts a raw .npy LiDAR depth map into a colormapped .png for FiftyOne visualization."""
    data = np.load(npy_path)
    x, y, z, mask = get_lidar_projections(data, azimuth, zenith)
    
    # Filter points using the mask 'mask == 1' (matching Nvidia's 3D scatter logic)
    x_scatter = x[mask == 1]
    z_scatter = z[mask == 1] # Z is the vertical height
    # y_scatter = y[mask == 1]
    
    # (Z-normalization)
    if len(z_scatter) > 0:
        c_min, c_max = np.min(z_scatter), np.max(z_scatter)
        if c_max > c_min:
            colors = (z_scatter - c_min) / (c_max - c_min)
        else:
            colors = z_scatter
    else:
        colors = z_scatter

    fig = plt.figure(figsize=(2, 2), dpi=128, facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    if len(x_scatter) > 0:
        ax.scatter(x_scatter, z_scatter, c=colors, cmap='inferno', s=5, edgecolors='none')

    # Adjust axes limit
    ax.set_xlim(-6, 6) 
    ax.set_ylim(-6, 6) 
    
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    return output_path

def get_torch_xyza(lidar_depth, azimuth, zenith):
    """Calculates 4-channel xyza from raw lidar depth."""
    # Ensure correct shapes for broadcasting
    x = lidar_depth * torch.sin(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    z = lidar_depth * torch.sin(-zenith[None, :])
    # Mask a: 1.0 for valid, 0.0 for background
    a = torch.where(lidar_depth < 50.0, torch.ones_like(lidar_depth), torch.zeros_like(lidar_depth))
    return torch.stack((x, y, z, a))

class CILPFusionDataset(Dataset):
    def __init__(self, root_dir, sample_ids=None, transform=None):
        """
        Args:
            root_dir (str): Path to the 'assessment' folder.
            sample_ids (list, optional): Specific list of IDs (e.g., ['0001', '1421']). 
                                         If None, all IDs in root_dir are discovered.
            transform (callable, optional): Optional transform to be applied on RGB.
        """
        self.root_dir = Path(os.path.expanduser(root_dir))
        self.samples = sample_ids
        self.transform = transform
        self.label_map = {"cube": 0, "sphere": 1}
        
        # Internal lookup tables
        self.id_to_class = {}
        self.angles = {}
        discovered_ids = []
        
        # We assume root_dir has 'cubes' and 'spheres' folders containing azymuth and zenith .npy files. Otherwise, adjust accordingly.
        for folder_name in ["cubes", "spheres"]:
            class_path = self.root_dir / folder_name
            if not class_path.exists():
                continue
            class_label = folder_name.rstrip('s') # 'cube' or 'sphere'
            # Load shared angles for this class
            self.angles[class_label] = {
                "azimuth": torch.from_numpy(np.load(class_path / "azimuth.npy")).to(torch.float32),
                "zenith": torch.from_numpy(np.load(class_path / "zenith.npy")).to(torch.float32)
            }
            # Scan for all RGB files to identify sample IDs
            rgb_dir = class_path / "rgb"
            if rgb_dir.exists():
                for f in rgb_dir.glob("*.png"):
                    stem = f.stem
                    self.id_to_class[stem] = class_label
                    discovered_ids.append(stem)
                    
        if sample_ids is not None:
            # Filter discovered data by provided list
            self.sample_ids = [s for s in sample_ids if s in self.id_to_class]
            if len(self.sample_ids) != len(sample_ids):
                missing = set(sample_ids) - set(self.id_to_class.keys())
                print(f"Warning: {len(missing)} provided IDs were not found in {root_dir}")
        else:
            # Use everything 
            self.sample_ids = sorted(discovered_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Retrieve the class for this ID
        class_label = self.id_to_class[sample_id]
        folder_name = class_label + "s"
        
        # Paths
        rgb_path = self.root_dir / folder_name / "rgb" / f"{sample_id}.png"
        lidar_path = self.root_dir / folder_name / "lidar" / f"{sample_id}.npy" 
        
        # Load RGB images
        rgb_img = Image.open(rgb_path).convert("RGBA")
        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        # Load LiDAR images 
        lidar_npy = np.load(lidar_path)
        lidar_depth = torch.from_numpy(lidar_npy).to(torch.float32)
        azi = self.angles[class_label]["azimuth"]
        zen = self.angles[class_label]["zenith"]
        lidar_xyza = get_torch_xyza(lidar_depth, azi, zen)
        
        label = torch.tensor(self.label_map[class_label], dtype=torch.long)
        return rgb_img, lidar_xyza, label