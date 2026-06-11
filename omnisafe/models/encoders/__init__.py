"""Encoders for omnisafe models."""

from omnisafe.models.encoders.nature_cnn import NatureCNN
from omnisafe.models.encoders.stacked_map_encoder import StackedMapEncoder

__all__ = ['NatureCNN', 'StackedMapEncoder']