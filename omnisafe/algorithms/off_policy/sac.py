# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Soft Actor-Critic algorithm."""

import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.world_model import TransformerWorldModel


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class SAC(DDPG):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    _log_alpha: torch.Tensor
    _alpha_optimizer: optim.Optimizer
    _target_entropy: float
    _world_model: TransformerWorldModel | None
    _world_model_optimizer: optim.Optimizer | None
    _use_world_model: bool

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()

        In SAC, we need to initialize the ``log_alpha`` and ``alpha_optimizer``.
        Additionally, we initialize the Transformer-based world model if enabled.
        """
        super()._init()
        
        # Initialize alpha (entropy temperature)
        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

            assert self._cfgs.model_cfgs.critic.lr is not None
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha],
                lr=self._cfgs.model_cfgs.critic.lr,
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device),
            )
        
        # Initialize Transformer-based world model (Think4CPP)
        self._use_world_model = False
        self._world_model = None
        self._world_model_optimizer = None
        self._world_model_update_counter = 0  # Counter to update world model every 16 steps
        
        if hasattr(self._cfgs.model_cfgs, 'world_model') and self._cfgs.model_cfgs.world_model.use_world_model:
            self._use_world_model = True
            wm_cfg = self._cfgs.model_cfgs.world_model
            
            # Get observation and action dimensions
            # Handle different observation space types
            obs_shape = getattr(self._env.observation_space, 'shape', None)
            if obs_shape is not None:
                if len(obs_shape) == 1:
                    obs_dim = obs_shape[0]
                else:
                    # For image observations, flatten
                    obs_dim = int(torch.prod(torch.tensor(obs_shape)).item())
            else:
                # Dict observation space - sum all dimensions
                obs_dim = sum(
                    int(torch.prod(torch.tensor(space.shape)).item())
                    for space in self._env.observation_space.spaces.values()
                )
            
            act_dim = self._env.action_space.shape[0]
            
            # Create world model
            self._world_model = TransformerWorldModel(
                obs_dim=obs_dim,
                act_dim=act_dim,
                embed_dim=wm_cfg.embed_dim,
                num_layers=wm_cfg.num_layers,
                num_heads=wm_cfg.num_heads,
                ffn_hidden_dim=wm_cfg.ffn_hidden_dim,
                dropout=wm_cfg.dropout,
            ).to(self._device)
            
            # Create optimizer for world model
            self._world_model_optimizer = optim.Adam(
                self._world_model.parameters(),
                lr=wm_cfg.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            print(f'World Model initialized with {obs_dim} obs_dim and {act_dim} act_dim')
            print(f'World Model architecture: {wm_cfg.num_layers} layers, {wm_cfg.num_heads} heads, {wm_cfg.embed_dim} embed_dim')

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Value/alpha')
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.register_key('Loss/alpha_loss')
        
        # Register world model logging keys if enabled
        if self._use_world_model:
            self._logger.register_key('world_model/total_loss')
            self._logger.register_key('world_model/nll_loss')
            self._logger.register_key('world_model/l2_loss')
            self._logger.register_key('world_model/kl_loss')
            self._logger.register_key('world_model/mean_variance')
            self._logger.register_key('world_model/mean_prediction_error')
            self._logger.register_key('world_model/mean_uncertainty')

    @property
    def _alpha(self) -> float:
        """The value of alpha."""
        return self._log_alpha.exp().item()
    
    def _normalize_obs(self, obs):
        """Normalize observation from uint8 [0, 255] to float [0, 1] range.
        
        Critical for world model training to prevent extremely high loss values.
        Observations are stored as uint8 in the environment/buffer but need to be
        normalized to [0, 1] for neural network training.
        
        Args:
            obs: Observation tensor (can be uint8 or float) or dict of tensors
            
        Returns:
            Normalized tensor in [0, 1] range as float32, or dict of normalized tensors
        """
        # Handle dict observations (e.g., fused image + lidar)
        if isinstance(obs, dict):
            return {key: self._normalize_obs(value) for key, value in obs.items()}
        
        # Convert to float if it's uint8
        if obs.dtype == torch.uint8:
            obs = obs.float()
        
        # Normalize to [0, 1] if values are in [0, 255] range
        if obs.max() > 1.0:
            obs = obs / 255.0
        
        return obs

    def _flatten_obs(self, obs) -> torch.Tensor:
        """Flatten observation tensor to 2D if needed.
        
        Args:
            obs: Observation tensor of shape (batch, ...) or (batch, seq, ...), or dict of tensors
            
        Returns:
            Flattened tensor of shape (batch, obs_dim) or (batch, seq, obs_dim)
        """
        # Handle dict observations (e.g., fused image + lidar) - flatten each and concatenate
        if isinstance(obs, dict):
            flattened_parts = []
            for key, value in obs.items():
                flattened_parts.append(self._flatten_obs(value))
            return torch.cat(flattened_parts, dim=-1)
        
        if obs.dim() == 2:
            # Already flat: (batch, obs_dim)
            return obs
        elif obs.dim() == 3:
            # Check if it's (batch, seq, obs_dim) or (batch, C, H) for images
            # If last dim is small and others are large, it's likely (batch, C, H, W) partially
            if obs.shape[-1] > 100 or obs.shape[-2] > 100:
                # Likely image-like, flatten last dims
                return obs.reshape(obs.shape[0], -1)
            else:
                # Likely already (batch, seq, obs_dim)
                return obs
        elif obs.dim() == 4:
            # Check if this is (batch, seq, H, W) or (batch, H, W, C)
            # If the 4th dim is small (channels), it's likely (batch, H, W, C)
            if obs.shape[-1] <= 16:  # Likely channels dimension
                # Image: (batch, H, W, C) -> (batch, H*W*C)
                return obs.reshape(obs.shape[0], -1)
            else:
                # Sequence of images: (batch, seq, C, H, W) -> (batch, seq, C*H*W)
                batch, seq = obs.shape[:2]
                return obs.reshape(batch, seq, -1)
        elif obs.dim() == 5:
            # Sequence of images: (batch, seq, H, W, C) -> (batch, seq, H*W*C)
            batch, seq = obs.shape[:2]
            return obs.reshape(batch, seq, -1)
        else:
            # Image or higher dim: flatten all but first dimension
            return obs.reshape(obs.shape[0], -1)

    def _train_world_model(self) -> None:
        """Train the Transformer-based world model on sequences from replay buffer.
        
        Implements the world model training procedure from Think4CPP paper:
        1. Sample sequences of length 64 from replay buffer
        2. Compute loss: NLL + λ₁*L2 + λ₂*LKL
        3. Update world model parameters
        """
        if not self._use_world_model or self._world_model is None:
            return
        
        wm_cfg = self._cfgs.model_cfgs.world_model
        
        # Check if we have enough data
        if self._buf._size < wm_cfg.start_learning_steps:
            return
        
        # Sample sequences from buffer
        sequences = self._buf.sample_sequences(
            seq_len=wm_cfg.sequence_length,
            batch_size=wm_cfg.batch_size,
        )
        
        if sequences is None:
            return
        
        # Extract and prepare data
        obs = sequences['obs']  # (batch, seq, ...)
        act = sequences['act']  # (batch, seq, act_dim)
        next_obs = sequences['next_obs']  # (batch, seq, ...)
        mask = sequences['mask']  # (batch, seq)
        
        # Normalize observations from uint8 [0, 255] to float [0, 1]
        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)
        
        # Flatten observations if needed
        obs_flat = self._flatten_obs(obs)
        next_obs_flat = self._flatten_obs(next_obs)
        
        # Train world model
        self._world_model.train()
        
        for _ in range(wm_cfg.update_iters):
            # Compute loss
            loss, loss_dict = self._world_model.compute_loss(
                states=obs_flat,
                actions=act,
                next_states=next_obs_flat,
                mask=mask,
                lambda_l2=wm_cfg.lambda_l2,
                lambda_kl=wm_cfg.lambda_kl,
            )
            
            # Update world model
            self._world_model_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (not explicitly mentioned in paper, but good practice)
            if self._cfgs.algo_cfgs.max_grad_norm:
                clip_grad_norm_(
                    self._world_model.parameters(),
                    self._cfgs.algo_cfgs.max_grad_norm,
                )
            
            self._world_model_optimizer.step()
            
            # Log world model metrics
            self._logger.store(loss_dict)
    
    def _get_world_model_uncertainty(
        self,
        obs,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get uncertainty estimate from world model for given state-action pairs.
        
        Args:
            obs: Current observation (batch, obs_dim) or dict of tensors
            action: Action to take (batch, act_dim)
            
        Returns:
            Uncertainty estimate (batch,) - scalar uncertainty per sample
        """
        if not self._use_world_model or self._world_model is None:
            # Get batch size from obs (handle dict or tensor)
            if isinstance(obs, dict):
                batch_size = next(iter(obs.values())).shape[0]
            else:
                batch_size = obs.shape[0]
            return torch.zeros(batch_size, device=self._device)
        
        self._world_model.eval()
        
        with torch.no_grad():
            # Normalize observation from uint8 [0, 255] to float [0, 1]
            obs = self._normalize_obs(obs)
            
            # Flatten observation if needed
            obs_flat = self._flatten_obs(obs)
            
            # Add sequence dimension (seq_len=1 for single step prediction)
            obs_seq = obs_flat.unsqueeze(1)  # (batch, 1, obs_dim)
            act_seq = action.unsqueeze(1)  # (batch, 1, act_dim)
            
            # Get uncertainty estimate
            _, _, uncertainty = self._world_model.predict_next_state(
                states=obs_seq,
                actions=act_seq,
            )
            
            # Remove sequence dimension
            uncertainty = uncertainty.squeeze(1)  # (batch,)
        
        return uncertainty

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Sample the target action by target actor.
        - Get the target Q value by target critic.
        - Use the minimum target Q value to update reward critic.
        - Add the entropy loss to reward critic.
        - Incorporate world model uncertainty into Q-target (Think4CPP).
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
            
            # Incorporate world model uncertainty into Q-target (Think4CPP Eq. 13)
            # Q_target = r + γ * [Q(s_future, a') - α * log π(a'|s_future) - β * U(s_future, a')]
            if self._use_world_model:
                wm_cfg = self._cfgs.model_cfgs.world_model
                uncertainty = self._get_world_model_uncertainty(next_obs, next_action)
                
                # Apply uncertainty penalty with beta weight
                next_q_value_r = next_q_value_r - wm_cfg.beta * uncertainty
                
                # Log mean uncertainty
                self._logger.store({
                    'world_model/mean_uncertainty': uncertainty.mean().item(),
                })
            
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, target_q_value_r) + nn.functional.mse_loss(
            q2_value_r,
            target_q_value_r,
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor and alpha if ``auto_alpha`` is True.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        super()._update_actor(obs)

        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                {
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                },
            )
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )
    
    def _update(self) -> None:
        """Update actor, critic, and world model.
        
        This method extends the DDPG update to include world model training (Think4CPP).
        
        Update order:
        1. Train world model on sequences (if enabled, every 16 steps)
        2. Update reward critic (with uncertainty-aware Q-target)
        3. Update cost critic
        4. Update actor policy
        """
        # Train world model first (Think4CPP) - only every 16th step
        if self._use_world_model:
            self._world_model_update_counter += 1
            if self._world_model_update_counter % 16 == 0:
                self._train_world_model()
        
        # Standard SAC updates (inherited from DDPG)
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)

            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SAC is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \alpha \log \pi (s)

        where :math:`Q^V` is the min value of two reward critic networks, and :math:`\pi` is the
        policy network, and :math:`\alpha` is the temperature parameter.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store(
                {
                    'Loss/alpha_loss': 0.0,
                },
            )
