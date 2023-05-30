from abc import ABCMeta, abstractmethod

import torch
from torch.nn import Module


class BaseEncoder(Module, metaclass=ABCMeta):
    """Base class of all encoders."""

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the given embeddings to a tensor.

        Args:
            embeddings: A [batch_size, sequence_length, embedding_size] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hidden_size(self) -> int:
        raise NotImplementedError
