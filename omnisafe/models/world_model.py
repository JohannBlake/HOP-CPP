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
"""Transformer-based World Model for Think4CPP.

This module implements the Transformer-based world model described in the Think4CPP paper.
The model predicts future states and their uncertainties using self-attention mechanisms
to capture long-range dependencies in coverage path planning tasks.

Reference:
    Think4CPP: Reinforcement Learning by Thinking with Latent World Model for Safe Coverage Path Planning
    Architecture: 6 Transformer encoder layers, 8 attention heads, embedding dimension 256
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer with modified scheme for spatial relationships.
    
    Uses sinusoidal encoding that preserves spatial relationships within coverage maps:
    PE(t)_i = sin(t/10000^(2i/d)) if i is even
    PE(t)_i = cos(t/10000^((2i-1)/d)) if i is odd
    
    Args:
        d_model: Embedding dimension (256 in paper)
        max_len: Maximum sequence length (default 5000)
        dropout: Dropout rate (0.1 in paper)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerWorldModel(nn.Module):
    """Transformer-based World Model for predicting future states with uncertainty.
    
    This model implements the architecture from Think4CPP paper:
    - 6 Transformer encoder layers
    - 8 attention heads per layer
    - Embedding dimension: 256
    - Position-wise FFN: 1024 hidden units
    - Outputs: Mean and log variance for Gaussian uncertainty modeling
    
    The model predicts p(s_{t+1}|s_t, a_t) = N(μ(s_t, a_t), σ²(s_t, a_t))
    
    Args:
        obs_dim: Observation space dimension
        act_dim: Action space dimension
        embed_dim: Embedding dimension (256 in paper)
        num_layers: Number of Transformer layers (6 in paper)
        num_heads: Number of attention heads (8 in paper)
        ffn_hidden_dim: Hidden dimension in position-wise FFN (1024 in paper)
        dropout: Dropout rate (0.1 in paper)
        max_seq_len: Maximum sequence length (default 5000)
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_hidden_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        
        # Joint state-action embedding (as per paper Eq. 8: Embedding(s, a))
        self.sa_embedding = nn.Linear(obs_dim + act_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output heads for mean and log variance
        # Two parallel fully-connected networks
        self.mean_head = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, obs_dim),
        )
        
        self.logvar_head = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, obs_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the world model.
        
        Args:
            states: State sequence tensor of shape (batch_size, seq_len, obs_dim)
            actions: Action sequence tensor of shape (batch_size, seq_len, act_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (predicted_mean, predicted_logvar) each of shape (batch_size, seq_len, obs_dim)
            - predicted_mean: Mean of predicted next state distribution
            - predicted_logvar: Log variance of predicted next state distribution
        """
        # Ensure inputs have sequence dimension
        if states.dim() == 2:
            states = states.unsqueeze(1)  # (batch, obs_dim) -> (batch, 1, obs_dim)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # (batch, act_dim) -> (batch, 1, act_dim)
        
        batch_size, seq_len, _ = states.shape
        
        # Joint embedding of state-action pairs (paper Eq. 8: Embedding(s, a))
        sa_concat = torch.cat([states, actions], dim=-1)  # (batch, seq, obs_dim + act_dim)
        sa_embed = self.sa_embedding(sa_concat)  # (batch, seq, embed_dim)
        
        # Add positional encoding (paper Eq. 8: Embedding(s, a) + PosEnc(t))
        sa_embed = self.pos_encoder(sa_embed)  # (batch, seq, embed_dim)
        
        # Apply Transformer encoder
        # Create attention mask if provided (convert to correct format)
        src_key_padding_mask = None
        if mask is not None:
            # mask: True for positions to ignore, False for positions to attend to
            src_key_padding_mask = ~mask.bool()
        
        encoded = self.transformer_encoder(
            sa_embed,
            src_key_padding_mask=src_key_padding_mask,
        )  # (batch, seq, embed_dim)
        
        # Generate mean and log variance predictions
        predicted_mean = self.mean_head(encoded)  # (batch, seq, obs_dim)
        predicted_logvar = self.logvar_head(encoded)  # (batch, seq, obs_dim)
        
        # Clamp log variance to prevent numerical instability
        predicted_logvar = torch.clamp(predicted_logvar, min=-10, max=2)
        
        return predicted_mean, predicted_logvar
    
    def predict_next_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the next state and its uncertainty.
        
        Args:
            states: State sequence tensor of shape (batch_size, seq_len, obs_dim)
            actions: Action sequence tensor of shape (batch_size, seq_len, act_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (next_state_mean, next_state_std, uncertainty) where:
            - next_state_mean: Mean prediction for next state (batch, seq, obs_dim)
            - next_state_std: Standard deviation (batch, seq, obs_dim)
            - uncertainty: Total uncertainty scalar per timestep (batch, seq)
        """
        mean, logvar = self.forward(states, actions, mask)
        std = torch.exp(0.5 * logvar)
        
        # Calculate average uncertainty across observation dimensions
        # Using mean instead of sum for stability and interpretability
        uncertainty = torch.mean(torch.exp(logvar), dim=-1)  # (batch, seq)
        
        return mean, std, uncertainty
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        lambda_l2: float = 1e-6,
        lambda_kl: float = 1e-3,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the world model loss as described in Think4CPP paper.
        
        Loss = NLL + λ₁*L2 + λ₂*LKL where:
        - NLL: Negative log-likelihood of Gaussian distribution
        - L2: Weight decay regularization
        - LKL: KL divergence penalty to prevent variance collapse
        
        Args:
            states: State sequence (batch, seq, obs_dim)
            actions: Action sequence (batch, seq, act_dim)
            next_states: Target next states (batch, seq, obs_dim)
            mask: Optional mask for valid timesteps
            lambda_l2: Weight for L2 regularization (1e-6 in paper)
            lambda_kl: Weight for KL divergence penalty (1e-3 in paper)
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual loss components
        """
        # Forward pass
        pred_mean, pred_logvar = self.forward(states, actions, mask)
        pred_var = torch.exp(pred_logvar)
        
        # Negative Log-Likelihood (NLL) loss
        # NLL = 0.5 * log(2π) + 0.5 * log(σ²) + (s' - μ)² / (2σ²)
        # Simplified: 0.5 * log(2πσ²) + (s' - μ)² / (2σ²)
        mse = (next_states - pred_mean) ** 2
        
        # Add small epsilon for numerical stability
        epsilon = 1e-6
        pred_var_stable = pred_var + epsilon
        
        # Compute NLL: 0.5 * [log(2πσ²) + (x-μ)²/σ²]
        log_2pi = math.log(2 * math.pi)
        nll = 0.5 * (log_2pi + pred_logvar + (mse / pred_var_stable))
        
        # Apply mask if provided
        if mask is not None:
            nll = nll * mask.unsqueeze(-1)
            nll = nll.sum() / (mask.sum() * next_states.shape[-1])  # Normalize by valid elements
        else:
            nll = nll.mean()
        
        # L2 regularization (weight decay)
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        l2_loss = lambda_l2 * l2_loss
        
        # KL divergence penalty to prevent variance collapse (full formula from paper)
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl_loss = 0.5 * torch.mean(pred_var + pred_mean ** 2 - 1 - pred_logvar)
        kl_loss = lambda_kl * kl_loss
        
        # Total loss
        total_loss = nll + l2_loss + kl_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'world_model/total_loss': total_loss.item(),
            'world_model/nll_loss': nll.item(),
            'world_model/l2_loss': l2_loss.item(),
            'world_model/kl_loss': kl_loss.item(),
            'world_model/mean_variance': pred_var.mean().item(),
            'world_model/mean_prediction_error': mse.mean().item(),
        }
        
        return total_loss, loss_dict
    
    def sample_next_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample next state from predicted distribution.
        
        Args:
            states: State sequence (batch, seq, obs_dim)
            actions: Action sequence (batch, seq, act_dim)
            mask: Optional mask
            
        Returns:
            Sampled next states (batch, seq, obs_dim)
        """
        mean, logvar = self.forward(states, actions, mask)
        std = torch.exp(0.5 * logvar)
        
        # Sample from Gaussian distribution
        eps = torch.randn_like(mean)
        sampled_states = mean + eps * std
        
        return sampled_states
