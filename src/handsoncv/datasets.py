import os 
import csv
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path

"""
The following functions are based on the notebooks provided for the Nvidia course 
Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

def get_lidar_xyza(lidar_depth, azimuth, zenith, return_stacked=True):
    """
    Universal LiDAR projection function for both NumPy and PyTorch.
    Args:
        lidar_depth: (H, W) array or tensor of depth values.
        azimuth: (H,) 1D array or tensor of horizontal angles.
        zenith: (W,) 1D array or tensor of vertical angles.
        return_stacked: If True, returns (4, H, W). If False, returns tuple of (x, y, z, a).
        
    Returns:
        Depending on return_stacked: 
        Stacked (4, H, W) object or tuple (x, y, z, mask).
    """
    # Auto-detect the backend
    is_torch = torch.is_tensor(lidar_depth)
    lib = torch if is_torch else np
    
    # Create the projections using broadcasting
    # Azimuth: rows ([:, None]), Zenith: columns ([None, :])
    x = lidar_depth * lib.sin(-azimuth[:, None]) * lib.cos(-zenith[None, :])
    y = lidar_depth * lib.cos(-azimuth[:, None]) * lib.cos(-zenith[None, :])
    z = lidar_depth * lib.sin(-zenith[None, :])
    
    # Create mask (a) for valid returns (max range 50.0 for this specific dataset; oterhwsie adjust accordingly)
    # This matches the 'a' in the Nvidia get_torch_xyza function
    mask = (lidar_depth < 50.0)
    
    # Cast mask to appropriate float type for the backend
    mask = mask.to(lidar_depth.dtype) if is_torch else mask.astype(np.float32)
    
    if return_stacked:
        return lib.stack([x, y, z, mask], dim=0 if is_torch else 0) # dim=0 for torch, axis=0 for numpy
    return x, y, z, mask

def create_lidar_viz(npy_path, azimuth, zenith, output_path):
    """Converts a raw .npy LiDAR depth map into a colormapped .png for FiftyOne visualization."""
    data = np.load(npy_path)
    x, y, z, mask = get_lidar_xyza(data, azimuth, zenith, return_stacked=False)
    
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

class CILPFusionDataset(Dataset):
    def __init__(self, root_dir, sample_ids=None, transform=None):
        """
        A Dataset for RGB-LiDAR fusion targeting binary classification (cube vs. sphere).
    
        Expected directory structure:
            root_dir/
                cubes/
                    rgb/*.png, lidar/*.npy, azimuth.npy, zenith.npy
                spheres/
                    rgb/*.png, lidar/*.npy, azimuth.npy, zenith.npy
                    
        Args:
            root_dir (str): Path to the 'assessment' folder.
            sample_ids (list, optional): Specific list of IDs (e.g., ['0001', '1421']). 
                                         If None, all IDs in root_dir are discovered.
            transform (callable, optional): Optional transform to be applied on RGB.
            
        Returns (via __getitem__):
            rgb_img (Tensor): The transformed image (typically 4-channel RGBA).
            lidar_xyza (Tensor): 4-channel LiDAR point data (XYZ + intensity/depth) 
                                computed from depth maps and spherical coordinates.
            label (LongTensor): Integer label (0 for cube, 1 for sphere).
        """
        self.root_dir = Path(os.path.expanduser(root_dir))
        self.samples = sample_ids
        self.transform = transform
        self.label_map = {"cube": 0, "sphere": 1}
        
        # Internal lookup tables
        self.angles = {}
        self.id_to_metadata = {} # unique IDs as keys
        discovered_unique_ids = []
        
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
                    # Create unique id: e.g., 'cube_0001'
                    unique_id = f"{class_label}_{f.stem}"
                    self.id_to_metadata[unique_id] = {
                        "stem": f.stem,
                        "class": class_label
                    }
                    discovered_unique_ids.append(unique_id)
                    
        if sample_ids is not None:
            # Filter our discovered unique_ids against the provided list
            self.sample_ids = [s for s in sample_ids if s in self.id_to_metadata]
            if len(self.sample_ids) == 0:
                print(f"Warning: IDs not found. Check .json file.")
        else:
            self.sample_ids = sorted(discovered_unique_ids)


    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        class_label, stem = sample_id.split("_") #e.g. 'cube_1538'
        folder_name = class_label + "s"
        
        # Paths
        rgb_path = self.root_dir / folder_name / "rgb" / f"{stem}.png"
        lidar_path = self.root_dir / folder_name / "lidar" / f"{stem}.npy" 
        
        # Load RGB images
        rgb_img = Image.open(rgb_path).convert("RGBA")
        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        # Load LiDAR images 
        lidar_npy = np.load(lidar_path)
        lidar_depth = torch.from_numpy(lidar_npy).to(torch.float32)
        azi = self.angles[class_label]["azimuth"]
        zen = self.angles[class_label]["zenith"]
        lidar_xyza = get_lidar_xyza(lidar_depth, azi, zen)
        
        label = torch.tensor(self.label_map[class_label], dtype=torch.long)
        return rgb_img, lidar_xyza, label

"""
The following functions are based on the modules provided for the Nvidia course
Generative AI with Diffusion Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-08+V1 
"""
    
def generate_clip_metadata(data_dir, save_path, clip_model, clip_preprocess, device):
    """One-time function to create the CSV with CLIP embeddings.
    
    Each row in the CSV contains:
        [image_path, clip_embedding_0, clip_embedding_1, ..., clip_embedding_N]
    """
    data_paths = glob.glob(f"{data_dir}/*/*.jpg")
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in data_paths:
            img = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                label = clip_model.encode_image(img)[0].cpu().tolist()
            writer.writerow([path] + label)
    print(f"Metadata saved to {save_path}")

class TFflowersCLIPDataset(Dataset):
    """
    PyTorch Dataset for loading cropped TF Flowers images paired with CLIP image 
    embeddings stored in a CSV file.

    Expected directory structure:
        data/cropped_flowers/
            <label_1>/
                <first_image_id>.jpg
                <second_image_id>.jpg
                ...
            <label_2>/
                <first_image_id>.jpg
                <second_image_id>jpg
                ...
            ...

    The CSV must contain one image per row with the following structure and is 
    is generated using a CLIP image encoder:
        image_path, emb_1, emb_2, ..., emb_D

        where:
        - `image_path`: Path to the corresponding image file
        - `emb_1 ... emb_D`: CLIP image embedding values (D = embedding dimension)

    In this project, the CSV file (e.g. `clip.csv`) is generated using a CLIP
    image encoder, as shown below:
    
    Args:
        csv_path (str): Path to the CSV file containing image paths and corresponding
            CLIP image embeddings.
        img_size (int, optional, default=32): Target spatial resolution to which images 
            are resized.
        transform (callable, optional): Optional transform to be applied on each image.

    Returns (via __getitem__):
        image (torch.FloatTensor): Transformed image of shape (3, img_size, img_size) 
        embedding (torch.FloatTensor): CLIP image embedding
    """
    def __init__(self, csv_path, img_size=32, transform=True):
        self.imgs_paths = []
        self.labels = []
        
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.imgs_paths.append(row[0])
                self.labels.append([float(x) for x in row[1:]])
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgs_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.imgs_paths)
    
class GeneratedMNISTDataset(Dataset):
    def __init__(self, results_list, transform=None):
        """
        results_list: the 'eval_results' list containing 'img_path' and 'condition'
        """
        # Only include results that actually saved an image
        self.samples = [r for r in results_list if r['img_path'] is not None]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]['img_path']
        label = self.samples[idx]['classifier_label'] #e.g. 0-9 integer or IDK
        
        # Load image (PNGs are saved as [0, 255] by save_image internally)
        image = Image.open(img_path).convert('L') # 'L' for grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label