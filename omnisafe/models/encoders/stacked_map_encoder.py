"""Stacked Map Features Extractor for fusing lidar and image observations.

This encoder is based on the StackedMapFeaturesExtractor architecture from architectures.py,
adapted for use with omnisafe. It fuses multi-scale map observations with lidar sensor data.
"""

import torch
import torch.nn as nn
import yaml
import os
from get_parameters import para


class StackedMapEncoder(nn.Module):
    """Encoder that fuses multi-scale map observations with lidar data.
    
    This encoder processes:
    - Multi-scale maps (coverage, obstacles, optionally frontier) through grouped convolutions
    - Lidar observations through a small MLP
    - Fuses both feature streams into a single feature vector
    
    Based on StackedMapFeaturesExtractor from stable-baselines3 implementation.
    """
    
    def __init__(
        self,
        image_channels: int,
        image_size: int,
        lidar_rays: int = 24,
        num_maps: int = 4,
        features_dim: int = 512,
        use_frontier: bool = True,
        grouped_convs: bool = True,
    ):
        """Initialize the StackedMapEncoder.
        
        Args:
            image_channels: Number of channels per map (coverage + obstacles + frontier)
            image_size: Size of input images (assumed square)
            lidar_rays: Number of lidar rays in the observation
            num_maps: Number of multi-scale maps (default 4)
            features_dim: Output feature dimension
            use_frontier: Whether frontier maps are included in observations
            grouped_convs: Whether to use grouped convolutions (one group per map)
        """
        super().__init__()
        
        # Load device configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']
        
        self.num_maps = num_maps
        self.use_frontier = use_frontier
        self.features_dim = features_dim
        self.lidar_rays = lidar_rays
        self.image_size = image_size
        
        # Calculate input channels: total channels from stacked maps
        # image_channels is the total channels (image_channels_per_map * num_maps)
        in_channels = image_channels
        out_channels = 2 * in_channels
        
        # Calculate output size after convolutions
        # After stride 2: image_size // 2
        # After 3 conv3x3 with stride 1: -2 each time
        out_size = (image_size // 2 - 2 - 2 - 2) ** 2 * out_channels
        
        if grouped_convs:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )
        else:
            self.map_extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(out_size, features_dim),
                nn.ReLU()
            )
        
        # Lidar feature extractor
        self.lidar_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lidar_rays, lidar_rays),
            nn.ReLU()
        )
        
        # Fused feature extractor
        self.fused_extractor = nn.Sequential(
            nn.Linear(features_dim + lidar_rays, features_dim),
            nn.ReLU()
        )
        
        # Move to device after initialization
        if 'cuda' in self.device and torch.cuda.is_available():
            self.to(self.device)
    
    def forward(self, image: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            image: Image tensor of shape (batch, H, W, C) or (batch, C, H, W) or (H, W, C)
            lidar: Lidar tensor of shape (batch, lidar_rays) or (lidar_rays,)
            
        Returns:
            Fused feature tensor of shape (batch, features_dim)
        """
        # Ensure input is on the same device as the model
        model_device = next(self.parameters()).device
        if image.device != model_device:
            image = image.to(model_device)
        if lidar.device != model_device:
            lidar = lidar.to(model_device)
        
        # Handle unbatched lidar input - add batch dimension if needed
        if lidar.dim() == 1:
            lidar = lidar.unsqueeze(0)
        
        # Handle image dimension ordering (batch, H, W, C) -> (batch, C, H, W)
        if image.dim() == 4 and image.shape[-1] != image.shape[-2]:
            # Assume last dimension is channels if not square
            image = image.permute(0, 3, 1, 2)
        elif image.dim() == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize image to [0, 1]
        image = image.float() / 255.0
        
        # Extract map features
        map_features = self.map_extractor(image)
        
        # Extract lidar features
        lidar_features = self.lidar_extractor(lidar.float())
        
        # Fuse features
        fused = torch.cat([map_features, lidar_features], dim=1)
        features = self.fused_extractor(fused)
        
        return features
