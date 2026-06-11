"""Fused Q Critic that combines lidar and image observations.

This critic uses the StackedMapEncoder to fuse multi-scale map observations
with lidar sensor data for Q-value estimation.
"""

import torch
import torch.nn as nn
import yaml
import os
from gymnasium import spaces
from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.models.encoders.stacked_map_encoder import StackedMapEncoder
from get_parameters import para


class FusedQCritic(Critic):
    """Fused CNN-Lidar Q Critic using StackedMapEncoder.
    
    This critic handles Dict observations containing both 'image' and 'lidar' keys,
    fusing them through the StackedMapEncoder before the Q-value network.
    """
    
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        features_dim: int = 512,
        hidden_sizes: list[int] = [256, 256],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
        lidar_rays: int = 24,
        num_maps: int = 4,
    ) -> None:
        # Don't call parent __init__ as it doesn't support Dict observation spaces
        # Instead, manually initialize nn.Module and set required attributes
        nn.Module.__init__(self)
        
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode = weight_initialization_mode
        self._activation = activation
        self._hidden_sizes = hidden_sizes
        self._num_critics = num_critics
        self._use_obs_encoder = use_obs_encoder
        
        # Get action dimension
        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim: int = act_space.shape[0]
        else:
            raise NotImplementedError("FusedQCritic only supports Box action spaces")
        
        # Dynamically load device configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']
        
        # Get observation dimensions
        if hasattr(obs_space, 'spaces'):
            # Dict observation space
            image_space = obs_space.spaces['image']
            lidar_space = obs_space.spaces['lidar']
            image_size = image_space.shape[0]
            image_channels = image_space.shape[2]
            lidar_rays = lidar_space.shape[0]
        else:
            # Fallback for Box space
            image_size = obs_space.shape[0]
            image_channels = obs_space.shape[2]
        
        print(f"FusedQCritic: image_size={image_size}, image_channels={image_channels}, lidar_rays={lidar_rays}")
        
        # Initialize encoder
        self.encoder = StackedMapEncoder(
            image_channels=image_channels,
            image_size=image_size,
            lidar_rays=lidar_rays,
            num_maps=num_maps,
            features_dim=features_dim,
            use_frontier=para.ablation_study_use_frontier_maps,
            grouped_convs=True,
        )
        
        # Initialize critic networks
        self.net_lst = []
        for idx in range(self._num_critics):
            net = nn.Sequential(
                nn.Linear(features_dim + self._act_dim, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], 1),
            )
            # Move critic network to device
            if 'cuda' in self.device and torch.cuda.is_available():
                net.to(self.device)
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)

    def _extract_observations(self, obs: torch.Tensor | dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract image and lidar from observations."""
        if isinstance(obs, dict):
            return obs['image'], obs['lidar']
        else:
            raise ValueError("FusedQCritic expects Dict observations with 'image' and 'lidar' keys")
        
    def forward(self, obs: torch.Tensor | dict, act: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the critic.
        
        Args:
            obs: Dict with 'image' and 'lidar' keys
            act: Action tensor
            
        Returns:
            List of Q-values from each critic
        """
        image, lidar = self._extract_observations(obs)
        
        # Get features from encoder
        features = self.encoder(image, lidar)
        
        # Ensure action is on the same device as features
        model_device = features.device
        if act.device != model_device:
            act = act.to(model_device)
        
        # Concatenate features and actions
        feature_action = torch.cat([features, act], dim=-1)
        
        res = []
        for net in self.net_lst:
            q_value = torch.squeeze(net(feature_action), -1)
            res.append(q_value)
        return res
