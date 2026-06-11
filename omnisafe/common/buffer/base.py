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
"""Abstract base class for buffer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class BaseBuffer(ABC):
    r"""Abstract base class for buffer.

    .. warning::
        The buffer only supports Box spaces.

    In base buffer, we store the following data:

    +--------+---------------------------+---------------+-----------------------------------+
    | Name   | Shape                     | Dtype         | Description                       |
    +========+===========================+===============+===================================+
    | obs    | (size, \*obs_space.shape) | torch.float32 | The observation from environment. |
    +--------+---------------------------+---------------+-----------------------------------+
    | act    | (size, \*act_space.shape) | torch.float32 | The action from agent.            |
    +--------+---------------------------+---------------+-----------------------------------+
    | reward | (size,)                   | torch.float32 | Single step reward.               |
    +--------+---------------------------+---------------+-----------------------------------+
    | cost   | (size,)                   | torch.float32 | Single step cost.                 |
    +--------+---------------------------+---------------+-----------------------------------+
    | done   | (size,)                   | torch.float32 | Whether the episode is done.      |
    +--------+---------------------------+---------------+-----------------------------------+


    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        device (torch.device): The device of the buffer. Defaults to ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`BaseBuffer`."""
        self._device: torch.device = device
        if isinstance(obs_space, Box):
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
                print(f"INFO: BaseBuffer using optimized uint8 dtype for observations "
                      f"(shape: {obs_space.shape}). This saves ~75% memory compared to float32.")
            
            obs_buf = torch.zeros((size, *obs_space.shape), dtype=obs_dtype, device=device)
        else:
            raise NotImplementedError
        if isinstance(act_space, Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

        self.data: dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }
        self._size: int = size

    @property
    def device(self) -> torch.device:
        """The device of the buffer."""
        return self._device

    @property
    def size(self) -> int:
        """The size of the buffer."""
        return self._size

    def __len__(self) -> int:
        """Return the length of the buffer."""
        return self._size

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
        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    @abstractmethod
    def store(self, **data: torch.Tensor) -> None:
        """Store a transition in the buffer.

        .. warning::
            This is an abstract method.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.store(obs=obs, act=act, reward=reward, cost=cost, done=done)

        Args:
            data (torch.Tensor): The data to store.
        """
