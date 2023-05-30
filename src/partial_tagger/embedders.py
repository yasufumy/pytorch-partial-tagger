from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from transformers import PreTrainedModel

from .data.batch import TaggerInputs


class BaseEmbedder(nn.Module, metaclass=ABCMeta):
    """Base class of all embedders."""

    @abstractmethod
    def forward(self, inputs: TaggerInputs) -> torch.Tensor:
        """Computes embeddings from the given inputs.

        Args:
            inputs: Any inputs feeding into an embedder.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, embedding_size] float tensor
            representing embeddings.
        """
        raise NotImplementedError


class TransformerEmbedder(BaseEmbedder):
    def __init__(self, model: PreTrainedModel):
        super(TransformerEmbedder, self).__init__()

        self.model = model

    def forward(self, inputs: TaggerInputs) -> torch.Tensor:
        return self.model(**inputs).last_hidden_state
