import torch
import torch.nn as nn
import yaml
import os
from torch.distributions import Normal
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.models.encoders.nature_cnn import NatureCNN
from omnisafe.utils.math import TanhNormal
from get_parameters import para


class CNNSACActor(Actor):
    """CNN-based SAC actor using NatureCNN as encoder for image observations.
    
    This actor outputs a stochastic policy suitable for SAC, combining:
    - NatureCNN encoder for processing image observations
    - Gaussian policy with learnable mean and log_std networks
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
    ) -> None:
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        # Dynamically load device configuration from algorithm-specific YAML file
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'off-policy', f'{para.omnisafe_alg}.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = config['defaults']['train_cfgs']['device']

        # Initialize encoder
        # obs_space.shape is (height, width, channels) for image observations
        image_size = obs_space.shape[0]  # Assuming square images (height == width)
        print("CNNSACActor obs_space.shape", obs_space.shape)
        self.encoder = NatureCNN(input_channels=obs_space.shape[2], features_dim=features_dim, image_size=image_size)
        
        # Policy network: outputs mean and log_std for each action dimension
        self.net = nn.Sequential(
            nn.Linear(features_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], self._act_dim * 2),  # *2 for mean and log_std
        )
        
        # Move networks to device after initialization (encoder already moved in its __init__)
        if 'cuda' in self.device and torch.cuda.is_available():
            self.net.to(self.device)
        
        self._current_raw_action: torch.Tensor | None = None
        self.register_buffer('_log2', torch.log(torch.tensor(2.0)))

    def _preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Preprocess observation: handle dimension ordering and normalization."""
        # obs: (batch, H, W, C) or (batch, C, H, W)
        if obs.dim() == 4 and obs.shape[-1] == self.encoder.cnn[0].in_channels:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.dim() == 3 and obs.shape[-1] == self.encoder.cnn[0].in_channels:
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        obs = obs.float() / 255.0
        return obs

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.
        
        Clips the standard deviation to a range of [-20, 2] for numerical stability.
        
        Args:
            obs (torch.Tensor): Observation from environments.
            
        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        obs = self._preprocess_obs(obs)
        features = self.encoder(obs)
        mean, log_std = self.net(features).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        return Normal(mean, std)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.
        
        The predicted action depends on the ``deterministic`` flag.
        
        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.
        
        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.
            
        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True

        action = self._current_dist.mean if deterministic else self._current_dist.rsample()

        self._current_raw_action = action

        return torch.tanh(action)

    def forward(self, obs: torch.Tensor) -> TanhNormal:
        """Forward method.
        
        Args:
            obs (torch.Tensor): Observation from environments.
            
        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return TanhNormal(self._current_dist.mean, self._current_dist.stddev)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.
        
        Warning:
            You must call forward() or predict() before calling this method.
            
        Note:
            We regularize the log probability for the tanh squashing:
            log prob = log π(a|s) - Σ(2 log 2 - a_i - log(1 + e^(-2 a_i)))
            
        Args:
            act (torch.Tensor): Action from predict() or forward().
            
        Returns:
            Log probability of the action.
        """
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
            ).sum(
                axis=-1,
            )
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
        raise NotImplementedError('CNNSACActor does not support setting std.')
