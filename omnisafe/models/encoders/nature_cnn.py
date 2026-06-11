import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from get_parameters import para

class NatureCNN(nn.Module):
    """Standard Nature CNN encoder for image observations."""
    def __init__(self, input_channels, features_dim=512, image_size=None):
        super().__init__()
        
        # Use provided image_size or fall back to para.observed_image_size
        if image_size is None:
            image_size = para.observed_image_size
        
        # Load device configuration dynamically
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']
        
        # Check if using multi-scale maps
        if not para.ablation_study_fisheye_instead_of_multi_scale_maps:
            # Multi-scale maps: input_channels contains 4 maps
            # Architecture similar to SimpleMapFeaturesExtractor with grouped convolutions
            num_maps = 4
            # Divide input_channels by num_maps to get channels per map
            channels_per_map = input_channels // num_maps
            
            # Group convolutions to process each map independently
            in_channels = input_channels
            out_channels = 2 * in_channels
            out_size = (image_size // 2 - 2 - 2 - 2)**2 * out_channels
            
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=num_maps),
                nn.ReLU(),
            )
            
            # Compute shape by doing one forward pass (on CPU first)
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_channels, image_size, image_size)
                n_flatten = self.cnn(dummy_input).view(1, -1).shape[1]
                
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        else:
            # Fisheye/single image: use standard Nature CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )
            
            # Compute shape by doing one forward pass (on CPU first)
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_channels, image_size, image_size)
                n_flatten = self.cnn(dummy_input).view(1, -1).shape[1]
                
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )
        
        # Move to device after initialization
        if 'cuda' in self.device and torch.cuda.is_available():
            self.to(self.device)

    def forward(self, x):
        # Ensure input is on the same device as the model
        model_device = next(self.parameters()).device
        if x.device != model_device:
            if not hasattr(self, '_device_move_logged'):
                self._device_move_logged = True
            x = x.to(model_device)
        
        x = self.cnn(x)
        x = self.linear(x)
        return x