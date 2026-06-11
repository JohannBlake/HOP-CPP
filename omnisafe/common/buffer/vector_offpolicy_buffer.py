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
"""Implementation of VectorOffPolicyBuffer."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Dict

from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class VectorOffPolicyBuffer(OffPolicyBuffer):
    """Vectorized on-policy buffer.

    The vector-off-policy buffer is a vectorized version of the off-policy buffer. It stores the
    data in a single tensor, and the data of each environment is stored in a separate column.

    .. warning::
        The buffer only supports Box spaces.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        num_envs (int): The number of environments.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOffPolicyBuffer`."""
        self._num_envs: int = num_envs
        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._device: torch.device = device
        self._is_dict_obs: bool = isinstance(obs_space, Dict)
        
        if isinstance(obs_space, Dict):
            # Handle Dict observation space (e.g., image + lidar)
            obs_buf = {}
            next_obs_buf = {}
            for key, space in obs_space.spaces.items():
                if isinstance(space, Box):
                    obs_dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
                    obs_buf[key] = torch.zeros(
                        (size, num_envs, *space.shape),
                        dtype=obs_dtype,
                        device=device,
                    )
                    next_obs_buf[key] = torch.zeros(
                        (size, num_envs, *space.shape),
                        dtype=obs_dtype,
                        device=device,
                    )
                    if space.dtype == np.uint8:
                        print(f"INFO: VectorOffPolicyBuffer using uint8 for obs['{key}'] "
                              f"(shape: {space.shape}). Saves ~75% memory.")
                else:
                    raise NotImplementedError(f"Dict obs space with non-Box subspace: {type(space)}")
        elif isinstance(obs_space, Box):
            # Use the observation space's dtype for optimal memory usage and consistency
            obs_dtype = torch.uint8 if obs_space.dtype == np.uint8 else torch.float32
            
            # IMPORTANT: Assertion for uint8 optimization awareness
            # This buffer has been optimized to automatically use uint8 when the observation
            # space uses uint8 (e.g., for image observations). This saves 75% memory.
            # If you're seeing unexpected dtypes, check the observation space definition.
            if obs_space.dtype == np.uint8:
                assert obs_dtype == torch.uint8, (
                    "CRITICAL: Buffer dtype logic error! Expected torch.uint8 for np.uint8 "
                    "observation space, but got different dtype. This indicates a bug in "
                    "the dtype conversion logic that needs to be fixed."
                )
                print(f"INFO: VectorOffPolicyBuffer using optimized uint8 dtype for observations "
                      f"(shape: {obs_space.shape}). This saves ~75% memory compared to float32.")
            
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=obs_dtype,
                device=device,
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=obs_dtype,
                device=device,
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self.data = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'cost': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'done': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'next_obs': next_obs_buf,
        }

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._num_envs

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape),
            dtype=dtype,
            device=self._device,
        )

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer, handling Dict observations.

        Args:
            data (torch.Tensor): The data to be stored.
        """
        for key, value in data.items():
            if key in ('obs', 'next_obs') and self._is_dict_obs:
                # Handle Dict observation
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.data[key][sub_key][self._ptr] = sub_value
                else:
                    # If value is a tensor but we expect dict, something is wrong
                    raise ValueError(f"Expected dict for {key} but got {type(value)}")
            else:
                self.data[key][self._ptr] = value
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        idx = torch.randint(
            0,
            self._size,
            (self._batch_size * self._num_envs,),
            device=self._device,
        )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._batch_size)
        
        if self._is_dict_obs:
            # Handle Dict observation space
            result = {}
            for key, value in self.data.items():
                if key in ('obs', 'next_obs') and isinstance(value, dict):
                    result[key] = {sub_key: sub_value[idx, env_idx] for sub_key, sub_value in value.items()}
                else:
                    result[key] = value[idx, env_idx]
            return result
        else:
            return {key: value[idx, env_idx] for key, value in self.data.items()}

    def sample_sequences(self, seq_len: int, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample sequences of consecutive transitions for world model training.
        
        This method samples contiguous sequences of length seq_len from the replay buffer.
        Used for training the Transformer-based world model in Think4CPP.
        
        Args:
            seq_len: Length of sequences to sample (64 in Think4CPP paper)
            batch_size: Number of sequences to sample (128 in Think4CPP paper)
            
        Returns:
            Dictionary containing:
                - obs: (batch_size, seq_len, obs_dim)
                - act: (batch_size, seq_len, act_dim)
                - next_obs: (batch_size, seq_len, obs_dim)
                - reward: (batch_size, seq_len)
                - done: (batch_size, seq_len)
                - mask: (batch_size, seq_len) - True for valid positions
        """
        if self._size < seq_len:
            # Not enough data yet, return empty tensors
            return None
        
        # Sample starting indices ensuring we have enough room for full sequences
        # Avoid crossing buffer boundaries
        max_start_idx = self._size - seq_len
        
        # Sample random starting indices
        start_indices = torch.randint(
            0,
            max_start_idx + 1,
            (batch_size,),
            device=self._device,
        )
        
        # Sample random environment indices
        env_indices = torch.randint(
            0,
            self._num_envs,
            (batch_size,),
            device=self._device,
        )
        
        # Create sequences
        sequences = {}
        for key in ['obs', 'act', 'reward', 'cost', 'done', 'next_obs']:
            if key in ('obs', 'next_obs') and self._is_dict_obs:
                # Handle Dict observation space - data[key] is a dict of tensors
                sequences[key] = {}
                for sub_key, sub_tensor in self.data[key].items():
                    seq_list = []
                    for i in range(batch_size):
                        start_idx = start_indices[i]
                        env_idx = env_indices[i]
                        seq = sub_tensor[start_idx:start_idx + seq_len, env_idx]
                        seq_list.append(seq)
                    sequences[key][sub_key] = torch.stack(seq_list, dim=0)
            else:
                # Regular tensor
                seq_list = []
                for i in range(batch_size):
                    start_idx = start_indices[i]
                    env_idx = env_indices[i]
                    # Extract sequence for this environment
                    seq = self.data[key][start_idx:start_idx + seq_len, env_idx]
                    seq_list.append(seq)
                # Stack into batch
                sequences[key] = torch.stack(seq_list, dim=0)  # (batch_size, seq_len, ...)
        
        # Create mask (all True since we sample valid sequences)
        # In more advanced implementations, this could mask out done transitions
        sequences['mask'] = torch.ones(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=self._device,
        )
        
        # Optionally mask out steps after episode termination
        # This makes training more realistic by not learning across episode boundaries
        for b in range(batch_size):
            done_indices = torch.where(sequences['done'][b])[0]
            if len(done_indices) > 0:
                first_done = done_indices[0].item()
                # Mask out everything after the first done
                sequences['mask'][b, first_done + 1:] = False
        
        return sequences
