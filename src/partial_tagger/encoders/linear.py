import torch
from torch import nn

from .base import BaseEncoder


class LinearEncoder(BaseEncoder):
    """A linear encoder.

    Args:
        embedding_size: An integer representing the embedding size.
        hidden_size:  An integer representing the hidden size.
    """

    def __init__(
        self, embedding_size: int, hidden_size: int, dropout: float = 0.2
    ) -> None:
        super(LinearEncoder, self).__init__()

        self.linear = nn.Linear(embedding_size, hidden_size)
        self.__hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes log potentials from the given embeddings.

        Args:
            embeddings: A [batch_size, sequence_length, embedding_size] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        return self.linear(self.dropout(embeddings))

    def get_hidden_size(self) -> int:
        return self.__hidden_size
