from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module

from ..data.core import LabelSet


class BaseEncoder(Module, metaclass=ABCMeta):
    """Base class of all encoders."""

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the given inputs to a tensor.

        Args:
            inputs: A dictionary that maps a string key to a tensor value.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hidden_size(self) -> int:
        raise NotImplementedError


class BaseEncoderFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, label_set: LabelSet) -> BaseEncoder:
        raise NotImplementedError
