import torch
import torch.nn as nn
import yaml
import os
from omnisafe.models.base import Critic
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.models.encoders.nature_cnn import NatureCNN
from get_parameters import para
class CNNQCritic(Critic):
    """CNN-based Q Critic using NatureCNN as encoder for image observations."""
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
    ) -> None:
        super().__init__(
            obs_space,
            act_space,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder,
        )
        
        # Dynamically load device configuration from algorithm-specific YAML file
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']
        
        # Initialize encoder (it will handle its own device placement)
        # obs_space.shape is (height, width, channels) for image observations
        image_size = obs_space.shape[0]  # Assuming square images (height == width)
        self.encoder = NatureCNN(input_channels=obs_space.shape[2], features_dim=features_dim, image_size=image_size)
        
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
            # Move critic network to the specified device
            if 'cuda' in self.device and torch.cuda.is_available():
                net.to(self.device)
            self.net_lst.append(net)
            self.add_module(f'critic_{idx}', net)
        
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> list[torch.Tensor]:
        # obs: (batch, H, W, C) or (batch, C, H, W)
        if obs.dim() == 4:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.dim() == 3:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        obs = obs.float() / 255.0
        
        # Debug only once per session
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
        
        # Get features from encoder (encoder handles device placement for obs)
        features = self.encoder(obs)
        
        # Ensure action is on the same device as features
        model_device = features.device
        if act.device != model_device:
            act = act.to(model_device)
        
        # Concatenate features and actions
        feature_action = torch.cat([features, act], dim=-1)
        
        res = []
        for i, net in enumerate(self.net_lst):
            q_value = torch.squeeze(net(feature_action), -1)
            res.append(q_value)
        return res