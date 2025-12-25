import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The following classes are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

class Embedder(nn.Module):
    """Reusable encoder backbone based on Nvidia 05_Assessment class."""
    def __init__(self, in_ch, downsample_mode='maxpool', return_vector=False, emb_dim=128):
        super().__init__()
        self.return_vector = return_vector 
        
        # Build layers dynamically to handle downsampling modes
        layers = []
        # Define the channel progression: in -> 50 -> 100 -> 200 -> 200
        channels = [in_ch, 50, 100, 200, 200]
        
        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            
            # Convolutional Block
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU()]
            
            # Downsampling Block
            if downsample_mode == 'maxpool':
                layers += [nn.MaxPool2d(2)]
            elif downsample_mode == 'stride':
                # Learns downsampling via strided convolution
                layers += [nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            else:
                raise ValueError(f"Downsample mode {downsample_mode} not implemented.")
        
        self.features = nn.Sequential(*layers) # output: [Batch, 200, 4, 4] for 64x64 input
                
        # If Late Fusion is used: we add a projection head to a vector space
        if self.return_vector:
            # Spatial size is 4x4 after 4 downsamplings of a 64x64 input (64 -> 32 -> 16 -> 8 -> 4)
            self.fc_emb = nn.Sequential(
                nn.Flatten(),
                nn.Linear(200 * 4 * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, emb_dim) # The low-dim bottleneck (e.g., 2)
            )
        
    def _get_downsample(self, ch, mode):
        if mode == 'maxpool':
            return [nn.MaxPool2d(2)]
        elif mode == 'stride':
            # Use strided convolution to downsample instead of pooling
            return [nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        else:
            raise ValueError(f"Downsample mode {mode} not supported.")

    def forward(self, x):
        x = self.features(x)
        if self.return_vector:
            x = self.fc_emb(x)
        return x

class LateFusionNet(nn.Module):
    """Late Fusion Net based on Nvidia 05_Assessment class."""
    def __init__(self, num_classes=2, emb_dim=128, downsample_mode='maxpool'):
        super().__init__()
        # return_vector=True makes the embedder spit out a 1D vector (embedding)
        self.rgb_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=True, emb_dim=emb_dim)
        self.lidar_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=True, emb_dim=emb_dim)
        
        # After convs and flattening: 200*4*4 = 3200
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 10),
            nn.ReLU(),
            nn.Linear(emb_dim * 10, num_classes)
        )

    def forward(self, rgb, lidar):
        e_rgb = self.rgb_encoder(rgb)
        e_lidar = self.lidar_encoder(lidar)
        combined = torch.cat((e_rgb, e_lidar), dim=1)
        return self.classifier(combined)

class IntermediateFusionNet(nn.Module):
    """Intermediate Fusion Net based on Nvidia 05_Assessment class with additional fusion strategies."""
    def __init__(self, mode='concat', num_classes=2, downsample_mode='maxpool'):
        super().__init__()
        self.mode = mode
        # return_vector=False keeps the 4D feature map [B, 200, 4, 4]
        self.rgb_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=False)
        self.lidar_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=False)
        
        # # Feature maps are 200x4x4 = 3200 elements per modality
        in_features = 3200 * 2 if mode == 'concat' else 3200 # add and mul preserve the channel dimension (200)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, lidar):
        # Outputs are feature maps [Batch, 200, 4, 4]
        f_rgb = self.rgb_encoder(rgb)
        f_lidar = self.lidar_encoder(lidar)
        
        if self.mode == 'concat':
            fused = torch.cat((f_rgb, f_lidar), dim=1)
        elif self.mode == 'add':
            fused = f_rgb + f_lidar
        elif self.mode == 'mul':
            fused = f_rgb * f_lidar # Hadamard product
            
        return self.classifier(fused)