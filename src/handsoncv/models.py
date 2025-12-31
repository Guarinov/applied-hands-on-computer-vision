import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
The following classes are based on the notebooks provided for the Nvidia course Building AI Agents with Multimodal Models https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-17+V1
"""

class Embedder(nn.Module):
    """
    Reusable encoder backbone based on Nvidia 05_Assessment class.
    CNN backbone reducing 64x64 inputs to 4x4 feature maps through 4 downsampling stages.
    
    Args:
        in_ch (int): Number of input channels (e.g., 4 for RGBA or XYZA).
        downsample_mode (str): 'maxpool' for fixed pooling or 'stride' for learned strided convs.
        return_vector (bool): If False, returns flattened 4x4 feature maps (B, emb_dim_interm * 16).
                             If True, applies a projection head to return a low-dim embedding.
        emb_dim_interm (int): Number of filters in the final convolutional layer (default 200).
        emb_dim_late (int): Dimension of the final bottleneck vector if return_vector is True.
    
    Returns:
        Tensor: (Batch, emb_dim_interm * 4 * 4) if return_vector=False.
        Tensor: (Batch, emb_dim_late) if return_vector=True.
    """
    def __init__(self, in_ch, downsample_mode='maxpool', return_vector=False, norm=False, emb_dim_interm=200, emb_dim_late=128):
        super().__init__()
        self.return_vector = return_vector 
        self.emb_dim_interm = emb_dim_interm
        self.emb_dim_late = emb_dim_late
        self.norm = norm
        
        # Build layers dynamically to handle downsampling modes
        layers = []
        # Define the channel progression: in -> 50 -> 100 -> 200 -> 200
        channels = [in_ch, 50, 100, 200, emb_dim_interm] #, 200] # Let's start with a smaller, less complex model
        
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
                # nn.Flatten(),
                nn.Linear(200 * 4 * 4, 128), #512 #nn.Linear(200 * 4 * 4, 512) if we add one more layer and make the Embedder deeper
                # nn.ReLU(),
                # nn.Linear(512, 128),
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
        x = x.flatten(1)
        if self.return_vector:
            x = self.fc_emb(x)
            if self.norm:
                return F.normalize(x)
        return x

class EmbedderStrided(Embedder):
    """
    Explicit class for Task 4. 
    
    Inherits from Embedder but forces strided convs.
    """
    def __init__(self, in_ch, return_vector=False, norm=False, emb_dim_interm=200, emb_dim_late=128):
        super().__init__(
            in_ch=in_ch, 
            downsample_mode='stride', 
            return_vector=return_vector, 
            emb_dim_interm=emb_dim_interm, 
            emb_dim_late=emb_dim_late,
            norm=norm,
        )

class LateFusionNet(nn.Module):
    """
    Late Fusion Net based on Nvidia 05_Assessment class.
    Two-stream network that fuses modalities at the bottleneck (embedding) level.
    
    Extracts 1D vectors (size: emb_dim_late) from both RGB and LiDAR streams, 
    concatenates them into a single vector (size: 2 * emb_dim_late), and 
    passes the result to a MLP classifier.
    """
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
    """
    Intermediate Fusion Net based on Nvidia 05_Assessment class with additional fusion strategies.
    Two-stream network that fuses modalities at the feature-map level (4x4 spatial grid).
    
    Args:
        mode (str): Fusion strategy. 
            'concat': Stacks features in the channel dimension (result: 2 * emb_dim_interm).
            'add': Element-wise summation of feature maps.
            'mul': Element-wise Hadamard product of feature maps.
    
    Note: 'add' and 'mul' preserve the channel dimension (emb_dim_interm), while 'concat' doubles it.
    """
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
        in_features = (emb_dim_interm * 4 * 4) * (2 if mode == 'concat' else 1) # add and mul preserve the channel dimension (200)
        # in_features = (emb_dim_interm * 8 * 8) * (2 if mode == 'concat' else 1) 
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, lidar):
        # Outputs are feature maps [Batch, 200 * 4 * 4]
        f_rgb = self.rgb_encoder(rgb)
        f_lidar = self.lidar_encoder(lidar)
        
        # Reshape to [Batch, 200, 4, 4] for applying intermediate fusion strategies
        f_rgb = f_rgb.view(f_rgb.size(0), 200, 4, 4)
        f_lidar = f_lidar.view(f_lidar.size(0), 200, 4, 4)
        
        if self.mode == 'concat':
            fused = torch.cat((f_rgb, f_lidar), dim=1)
        elif self.mode == 'add':
            fused = f_rgb + f_lidar
        elif self.mode == 'mul':
            fused = f_rgb * f_lidar # Hadamard product
            
        return self.classifier(fused)
    
class LidarClassifier(nn.Module):
    """
    LiDAR classifier adapted from NVIDIA’s 05_Assessment class. 
    Classification head designed to process LiDAR features. 
    
    It can either process raw 4-channel LiDAR data through its internal Embedder 
    or accept pre-computed intermediate features (f) directly. This flexibility 
    is used for modular fine-tuning and cross-modal projection tasks.
    """
    def __init__(self, emb_dim_interm=200, num_classes=2):
        super().__init__()
        self.embedder = Embedder(in_ch=4, downsample_mode='maxpool', return_vector=False, emb_dim_interm=emb_dim_interm) # best 4-channel Embedder based on Task 3 & Task 4
        
        # Classifier head for 8x8 spatial output (Modify it to 4x4 if an additional convolution layer is added to the Embedder
        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(emb_dim_interm * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, raw_data=None, return_embs=False, f=None):
        if raw_data is not None:
            f = self.embedder(raw_data)
            if return_embs:
                return f #.flatten(1) # Return the [B, 12800] embedding
        return self.classifier(f)

class EfficientCILPModel(nn.Module):
    """
    Contrastive Image-LiDAR Pre-training (CILP) model using dot-product similarity.
    
    Encodes both modalities into normalized 1D vectors and computes a (Batch x Batch) 
    similarity matrix. Uses a learnable temperature parameter (logit_scale) to 
    "sharpen" the distribution, following the CLIP architecture.
    """
    def __init__(self, emb_dim_interm=200, emb_dim_late=200):
        super().__init__()
        # Return_vector=True gives us the [B, 128] bottleneck
        self.img_embedder = Embedder(4, return_vector=True, norm=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.lidar_embedder = Embedder(4, return_vector=True, norm=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, rgb, lidar):
        img_emb = self.img_embedder(rgb)
        lidar_emb = self.lidar_embedder(lidar)
        
        # Matrix of similarities [Batch, Batch]
        logits = img_emb @ lidar_emb.t() * self.logit_scale.exp() # Cosine similarity ((a * b)/(||a||*||b||)) scaled with learnable temperature
        return logits

class CILPModel(nn.Module):
    """
    Contrastive Pretraining Model for Image-LiDAR embeddings Pairs (CILP) based on Nvidia 05_Assessment class.
    
    Encodes both modalities into normalized 1D vectors and computes a (Batch x Batch) 
    similarity matrix. Uses a fixed scaling for normalizing the distribution.
    """
    def __init__(self, emb_dim_interm=200, emb_dim_late=200):
        super().__init__()
        # Return_vector=True gives us the [B, 128] bottleneck
        self.img_embedder = Embedder(4, return_vector=True, norm=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.lidar_embedder = Embedder(4, return_vector=True, norm=True, emb_dim_interm=emb_dim_interm, emb_dim_late=emb_dim_late)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, rgb_imgs, lidar_depths):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_depths)
        batch_size = img_emb.shape[0]

        # Resulting shape: [batch_size * batch_size, emb_dim]
        repeated_img_emb = img_emb.repeat_interleave(batch_size, dim=0)
        repeated_lidar_emb = lidar_emb.repeat(batch_size, 1)

        similarity_logits = self.cos(repeated_img_emb, repeated_lidar_emb) # Compute similarity
        similarity_logits = torch.unflatten(similarity_logits, 0, (batch_size, batch_size)) # Reshape into [Batch, Batch] matrix
        similarity_logits = (similarity_logits + 1) / 2 # Nvidia-specific scaling

        return similarity_logits

class CrossModalProjector(nn.Module):
    """
    A multi-layer perceptron (MLP) that maps a compressed RGB late embedding 
    (e.g., dim 128) back into the high-dimensional LiDAR feature space (e.g., dim 3200).
    
    This acts as the 'bridge' in the RGB2LiDAR pipeline, connecting an RGB encoder to 
    a classifier originally trained on LiDAR features.
    """
    def __init__(self, rgb_dim=200, lidar_dim=3200): #12800 if using 8x8 feature maps; 3200 if using 4x4 feature maps
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rgb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 512), #4096
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # nn.Linear(512, lidar_dim),
            nn.Linear(2048, lidar_dim),
        )

    def forward(self, x):
        return self.net(x)

class RGB2LiDARClassifier(nn.Module):
    """
    Fine-Tuning inspired by "Visual Instruction Tuning": RGB input -> CILP RGB Enc -> Projector -> -> LiDAR Head.
    
    Input: (B, 4, 64, 64) RGB image.
    Output: (B, num_classes) logits.
    """
    def __init__(self, rgb_enc, projector, lidar_classifier):
        super().__init__()
        self.rgb_enc = rgb_enc # trained CILP RGB Embedder
        self.projector = projector # trained CrossModalProjector
        self.lidar_classifier = lidar_classifier # trained LidarClassifier

    def forward(self, x):
        img_emb = self.rgb_enc(x) # RGB Encoder trained with contrastive pretraining 
        proj_lidar_emb = self.projector(img_emb) #Flattened to match [B, 200, 8, 8] or [B, 200, 4, 4] expected by LiDAR's classifier head
        return self.lidar_classifier(f=proj_lidar_emb)