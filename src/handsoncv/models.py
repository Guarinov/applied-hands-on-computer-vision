import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
The following classes are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

class Embedder(nn.Module):
    """Reusable encoder backbone based on Nvidia 05_Assessment class."""
    def __init__(self, in_ch, downsample_mode='maxpool', return_vector=False, emb_dim_interm=200, emb_dim_late=128):
        super().__init__()
        self.return_vector = return_vector 
        self.emb_dim_interm = emb_dim_interm
        self.emb_dim_late = emb_dim_late
        
        # Build layers dynamically to handle downsampling modes
        layers = []
        # Define the channel progression: in -> 50 -> 100 -> 200 -> 200
        channels = [in_ch, 50, 100, emb_dim_interm] #, 200] # Let's start with a smaller, less complex model
        
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
                nn.Linear(200 * 8 * 8, 512), #nn.Linear(200 * 4 * 4, 512) if we add one more layer and make the Embedder deeper
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, emb_dim_late) # The low-dim bottleneck (e.g., 2)
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

class EmbedderStrided(Embedder):
    """Explicit class for Task 4. Inherits from Embedder but forces strided convs."""
    def __init__(self, in_ch, return_vector=False, emb_dim_interm=200, emb_dim_late=128):
        super().__init__(
            in_ch=in_ch, 
            downsample_mode='stride', 
            return_vector=return_vector, 
            emb_dim_interm=emb_dim_interm, 
            emb_dim_late=emb_dim_late
        )

class LateFusionNet(nn.Module):
    """Late Fusion Net based on Nvidia 05_Assessment class."""
    def __init__(self, num_classes=2, emb_dim_interm=200, emb_dim_late=128, downsample_mode='maxpool'):
        super().__init__()
        # return_vector=True makes the embedder spit out a 1D vector (embedding)
        if downsample_mode == 'stride':
            self.rgb_encoder = Embedder(4, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
            self.lidar_encoder = Embedder(4, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        else:
            self.rgb_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
            self.lidar_encoder = Embedder(4, downsample_mode=downsample_mode, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        
        # After convs and flattening: 200*4*4 = 3200
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim_late * 2, emb_dim_late * 10),
            nn.ReLU(),
            nn.Linear(emb_dim_late * 10, num_classes)
        )

    def forward(self, rgb, lidar):
        e_rgb = self.rgb_encoder(rgb)
        e_lidar = self.lidar_encoder(lidar)
        combined = torch.cat((e_rgb, e_lidar), dim=1)
        return self.classifier(combined)

class IntermediateFusionNet(nn.Module):
    """Intermediate Fusion Net based on Nvidia 05_Assessment class with additional fusion strategies."""
    def __init__(self, mode='concat', num_classes=2, emb_dim_interm=200, downsample_mode='maxpool'):
        super().__init__()
        self.mode = mode
        # return_vector=False keeps the 4D feature map [B, 200, 4, 4]
        if downsample_mode == 'stride':
            self.rgb_encoder = EmbedderStrided(4, return_vector=False, emb_dim_interm=emb_dim_interm)
            self.lidar_encoder = EmbedderStrided(4, return_vector=False, emb_dim_interm=emb_dim_interm)
        else:
            self.rgb_encoder = Embedder(4, downsample_mode='maxpool', return_vector=False, emb_dim_interm=emb_dim_interm)
            self.lidar_encoder = Embedder(4, downsample_mode='maxpool', return_vector=False, emb_dim_interm=emb_dim_interm)
        
        # # Feature maps are 200x4x4 = 3200 elements per modality
        # Feature maps dimensiones modified for lighter embedder : 200x8x8 = 12800 elements per modality
        in_features = (emb_dim_interm * 8 * 8) * (2 if mode == 'concat' else 1) # add and mul preserve the channel dimension (200)
        # in_features = (emb_dim_interm * 4 * 4) * (2 if mode == 'concat' else 1) 
        
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
    
class LidarClassifier(nn.Module):
    """LiDAR classifier adapted from NVIDIA’s 05_Assessment class.  
    Designed for fine-tuning after training the projection module and refining feature embeddings via contrastive pretraining."""
    def __init__(self, emb_dim_interm=200, num_classes=2):
        super().__init__()
        self.embedder = Embedder(in_ch=4, downsample_mode='maxpool', return_vector=False, emb_dim_interm=emb_dim_interm) # best 4-channel Embedder based on Task 3 & Task 4
        
        # Classifier head for 8x8 spatial output (Modify it to 4x4 if an additional convolution layer is added to the Embedder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim_interm * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_embs=False, f=None):
        if x is not None:
            f = self.embedder(x)
            if return_embs:
                return f.flatten(1) # Return the [B, 12800] embedding
        return self.classifier(f)

class CILPModel(nn.Module):
    """Contrastive Pretraining Model for Image-LiDAR embeddings Pairs (CILP) based on Nvidia 05_Assessment class."""
    def __init__(self, emb_dim_interm=200, emb_dim_late=200):
        super().__init__()
        # return_vector=True gives us the [B, 128] bottleneck
        self.img_embedder = Embedder(4, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.lidar_embedder = Embedder(4, return_vector=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, rgb, lidar):
        img_emb = self.img_embedder(rgb)
        lidar_emb = self.lidar_embedder(lidar)
        
        # Normalize for cosine similarity
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        lidar_emb = lidar_emb / lidar_emb.norm(dim=-1, keepdim=True)
        
        # Matrix of similarities [Batch, Batch]
        logits = img_emb @ lidar_emb.t() * self.logit_scale.exp()
        return logits

class CrossModalProjector(nn.Module):
    """Map RGB late embedding to LiDAR intermediate space."""
    def __init__(self, rgb_dim=200, lidar_dim=12800):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rgb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, lidar_dim)
        )

    def forward(self, x):
        return self.net(x)

class RGB2LiDARClassifier(nn.Module):
    """Fine-Tuning inspired by "Visual Instruction Tuning": RGB input -> CILP RGB Enc -> Projector -> Lidar Classifier Head."""
    def __init__(self, rgb_enc, projector, lidar_classifier):
        super().__init__()
        self.rgb_enc = rgb_enc
        self.projector = projector # trained CrossModalProjector
        self.lidar_classifier = lidar_classifier # trained LidarClassifier

    def forward(self, x):
        with torch.no_grad():
            img_emb = self.rgb_enc(x) # RGB Encoder trained with contrastive pretraining 
        proj_lidar_emb = self.projector(img_emb) #Flattened to match [B, 200, 8, 8] expected by LiDAR's classifier head
        return self.lidar_classifier(f=proj_lidar_emb)