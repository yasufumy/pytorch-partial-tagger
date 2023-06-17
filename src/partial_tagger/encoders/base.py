from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module

from partial_tagger.data.core import LabelSet


class BaseEncoder(Module, metaclass=ABCMeta):
    """Base class for all encoders."""

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes the given inputs to a tensor representation.

        Args:
            inputs: A dictionary that maps string keys to a tensor values.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hidden_size(self) -> int:
        """Returns the dimension size of the output tensor.

        Returns:
            The dimension size of the output tensor.
        """
        raise NotImplementedError


class BaseEncoderFactory(metaclass=ABCMeta):
    """Base class for all encoder factories."""

    @abstractmethod
    def create(self, label_set: LabelSet) -> BaseEncoder:
        """Creates an encoder based on the provided label set.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            An encoder that transforms input into a tensor representation.

        """
        raise NotImplementedError
