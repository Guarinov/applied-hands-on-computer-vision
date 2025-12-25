import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The following classes are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

class CNNBlock(nn.Module):
    """Reusable encoder backbone based on your Nvidia 05_Assessment class."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # output: [Batch, 200, 4, 4] for 64x64 input

    def forward(self, x):
        return self.conv(x)

class LateFusionNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_encoder = CNNBlock(4)
        self.lidar_encoder = CNNBlock(4)
        
        # After convs and flattening: 200*4*4 = 3200
        self.fc = nn.Sequential(
            nn.Linear(3200 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, lidar):
        x1 = torch.flatten(self.rgb_encoder(rgb), 1)
        x2 = torch.flatten(self.lidar_encoder(lidar), 1)
        combined = torch.cat((x1, x2), dim=1)
        return self.fc(combined)

class IntermediateFusionNet(nn.Module):
    def __init__(self, mode='concat', num_classes=2):
        super().__init__()
        self.mode = mode
        self.rgb_encoder = CNNBlock(4)
        self.lidar_encoder = CNNBlock(4)
        
        # Fusion logic
        in_features = 3200 * 2 if mode == 'concat' else 3200
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, lidar):
        f_rgb = self.rgb_encoder(rgb)
        f_lidar = self.lidar_encoder(lidar)
        
        if self.mode == 'concat':
            fused = torch.cat((f_rgb, f_lidar), dim=1)
        elif self.mode == 'add':
            fused = f_rgb + f_lidar
        elif self.mode == 'mul':
            fused = f_rgb * f_lidar # Hadamard product
            
        return self.fc(torch.flatten(fused, 1))