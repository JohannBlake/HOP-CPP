"""Fused SAC Actor that combines lidar and image observations.

This actor uses the StackedMapEncoder to fuse multi-scale map observations
with lidar sensor data for the SAC algorithm.
"""

import torch
import torch.nn as nn
import yaml
import os
from gymnasium import spaces
from torch.distributions import Normal
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.models.encoders.stacked_map_encoder import StackedMapEncoder
from omnisafe.utils.math import TanhNormal
from get_parameters import para


class FusedSACActor(Actor):
    """Fused CNN-Lidar SAC actor using StackedMapEncoder.
    
    This actor handles Dict observations containing both 'image' and 'lidar' keys,
    fusing them through the StackedMapEncoder before the policy network.
    """
    
    _log2: torch.Tensor
    _current_dist: Normal
    
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        features_dim: int = 512,
        hidden_sizes: list[int] = [256, 256],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
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
        self._after_inference = False
        
        # Get action dimension
        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim: int = act_space.shape[0]
        else:
            raise NotImplementedError("FusedSACActor only supports Box action spaces")

        # Dynamically load device configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']

        # Get image observation dimensions
        # obs_space is a Dict space with 'image' and 'lidar' keys
        if hasattr(obs_space, 'spaces'):
            # Dict observation space
            image_space = obs_space.spaces['image']
            lidar_space = obs_space.spaces['lidar']
            image_size = image_space.shape[0]
            image_channels = image_space.shape[2]
            lidar_rays = lidar_space.shape[0]
        else:
            # Fallback for Box space (shouldn't happen with fusion)
            image_size = obs_space.shape[0]
            image_channels = obs_space.shape[2]
        
        print(f"FusedSACActor: image_size={image_size}, image_channels={image_channels}, lidar_rays={lidar_rays}")
        
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
        
        # Policy network: outputs mean and log_std for each action dimension
        self.net = nn.Sequential(
            nn.Linear(features_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], self._act_dim * 2),  # *2 for mean and log_std
        )
        
        # Move networks to device
        if 'cuda' in self.device and torch.cuda.is_available():
            self.net.to(self.device)
        
        self._current_raw_action: torch.Tensor | None = None
        self.register_buffer('_log2', torch.log(torch.tensor(2.0)))

    def _extract_observations(self, obs: torch.Tensor | dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract image and lidar from observations.
        
        Args:
            obs: Either a dict with 'image' and 'lidar' keys, or a flattened tensor
            
        Returns:
            Tuple of (image, lidar) tensors
        """
        if isinstance(obs, dict):
            return obs['image'], obs['lidar']
        else:
            # If obs is a tensor, it should be the concatenation of image (flattened) and lidar
            # This shouldn't happen in normal usage but handle it gracefully
            raise ValueError("FusedSACActor expects Dict observations with 'image' and 'lidar' keys")

    def _distribution(self, obs: torch.Tensor | dict) -> Normal:
        """Get the distribution of the actor."""
        image, lidar = self._extract_observations(obs)
        features = self.encoder(image, lidar)
        mean, log_std = self.net(features).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        return Normal(mean, std)

    def predict(self, obs: torch.Tensor | dict, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation."""
        self._current_dist = self._distribution(obs)
        self._after_inference = True

        action = self._current_dist.mean if deterministic else self._current_dist.rsample()
        self._current_raw_action = action
        return torch.tanh(action)

    def forward(self, obs: torch.Tensor | dict) -> TanhNormal:
        """Forward method."""
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return TanhNormal(self._current_dist.mean, self._current_dist.stddev)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action."""
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False

        if self._current_raw_action is not None:
            logp = self._current_dist.log_prob(self._current_raw_action).sum(axis=-1)
            logp -= (
                2
                * (
                    self._log2
                    - self._current_raw_action
                    - nn.functional.softplus(
                        -2 * self._current_raw_action,
                    )
                )
            ).sum(axis=-1)
            self._current_raw_action = None
        else:
            logp = (
                TanhNormal(self._current_dist.mean, self._current_dist.stddev)
                .log_prob(act)
                .sum(axis=-1)
            )

        return logp

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._current_dist.stddev.mean().item()

    @std.setter
    def std(self, std: float) -> None:
        raise NotImplementedError('FusedSACActor does not support setting std.')
