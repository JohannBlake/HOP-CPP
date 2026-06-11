import torch
import torch.nn as nn
import yaml
import os
from torch.distributions import Normal, Distribution
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.models.encoders.nature_cnn import NatureCNN
from get_parameters import para

class CNNActor(Actor):
    """CNN-based actor using NatureCNN as encoder for image observations."""
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        features_dim: int = 512,
        hidden_sizes: list[int] = [256, 256],
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        # Dynamically load device configuration from algorithm-specific YAML file
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']

        # Initialize encoder and actor network
        # obs_space.shape is (height, width, channels) for image observations
        image_size = obs_space.shape[0]  # Assuming square images (height == width)
        print("obs_space.shape", obs_space.shape)
        self.encoder = NatureCNN(input_channels=obs_space.shape[2], features_dim=features_dim, image_size=image_size)
        self.net = nn.Sequential(
            nn.Linear(features_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], self._act_dim),
            nn.Tanh(),
        )
        
        # Move networks to device after initialization (encoder already moved in its __init__)
        if 'cuda' in self.device and torch.cuda.is_available():
            self.net.to(self.device)
        
        
        self._noise = 0.1

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        # For compatibility, output a Normal distribution with fixed std
        mu = self._forward_net(obs)
        std = torch.ones_like(mu) * self._noise
        return Normal(mu, std)

    def forward(self, obs: torch.Tensor) -> Distribution:
        return self._distribution(obs)

    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action = self._forward_net(obs)
        if deterministic:
            return action
        with torch.no_grad():
            noise = torch.normal(0, self._noise * torch.ones_like(action))
            return torch.clamp(action + noise, -1, 1)

    @property
    def noise(self) -> float:
        """Noise of the action."""
        return self._noise

    @noise.setter
    def noise(self, noise: float) -> None:
        """Set the action noise."""
        assert noise >= 0, 'Noise should be non-negative.'
        self._noise = noise

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        # Not used for deterministic policies, but required for interface
        # Assume mean 0, std self._noise for all actions
        mu = torch.zeros_like(act)
        std = torch.ones_like(act) * self._noise
        dist = Normal(mu, std)
        return dist.log_prob(act).sum(-1)

    def _forward_net(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, H, W, C) or (batch, C, H, W)
        if obs.dim() == 4 and obs.shape[-1] == self.encoder.cnn[0].in_channels:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.dim() == 3 and obs.shape[-1] == self.encoder.cnn[0].in_channels:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        obs = obs.float() / 255.0
        
        # Debug only once per session
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
        
        features = self.encoder(obs)
        action = self.net(features)
        
        return action