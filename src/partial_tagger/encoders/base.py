from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module

from ..data.batch.text import TaggerInputs


class BaseEncoder(Module, metaclass=ABCMeta):
    """Base class of all encoders."""

    @abstractmethod
    def forward(self, inputs: TaggerInputs) -> torch.Tensor:
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
