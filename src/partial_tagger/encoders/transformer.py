from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel

from partial_tagger.data.core import LabelSet
from partial_tagger.encoders.base import BaseEncoder, BaseEncoderFactory


class TransformerModelEncoder(BaseEncoder):
    """A Transformer-based encoder for transforming inputs into
    a fixed-size tensor representation.

    Args:
        model: A pre-trained transformer model to use for encoding.
        embedding_size: An integer representing the size of the input embeddings.
        hidden_size: An integer representing the dimension size of
            the output tensor representation.
        dropout_prob: A float representing dropout probability to apply.
            Defaults to 0.2.

    Attributes:
        model: A pre-trained transformer model.
        linear: A linear layer for projecting embeddings to the hidden size.
        dropout: A dropout layer for regularization.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        embedding_size: int,
        hidden_size: int,
        dropout_prob: float = 0.2,
    ):
        super(TransformerModelEncoder, self).__init__()

        self.model = model
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.__hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes the given inputs to a tensor representation.

        Args:
            inputs: A dictionary that maps string keys to a tensor values.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        embeddings = self.model(**inputs).last_hidden_state
        return self.linear(self.dropout(embeddings))

    def get_hidden_size(self) -> int:
        """Returns the dimension size of the output tensor.

        Returns:
            The dimension size of the output tensor.
        """
        return self.__hidden_size


class TransformerModelEncoderFactory(BaseEncoderFactory):
    """Factory class for creating TransformerModelEncoder instances.

    Args:
        model_name: The name or path of the pre-trained transformer model.
        dropout_prob: Dropout probability to apply. Defaults to 0.2.
    """

    def __init__(self, model_name: str, dropout_prob: float = 0.2):
        self.__model_name = model_name
        self.__dropout_prob = dropout_prob

    def create(self, label_set: LabelSet) -> TransformerModelEncoder:
        """Creates an TransformerModelEncoder instance based on the provided label set.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            An encoder that transforms input into a tensor representation.
        """
        model = AutoModel.from_pretrained(self.__model_name)
        return TransformerModelEncoder(
            model,
            model.config.hidden_size,
            label_set.get_tag_size(),
            self.__dropout_prob,
        )


class TransformerModelWithHeadEncoder(BaseEncoder):
    """A Transformer-based encoder for transforming inputs into
    a fixed-size tensor representation.

    Args:
        model: A transformer model with a classification head.

    Attributes:
        model: A pre-trained transformer model.
    """

    def __init__(self, model: AutoModelForTokenClassification):
        super(TransformerModelWithHeadEncoder, self).__init__()

        self.model = model

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes the given inputs to a tensor representation.

        Args:
            inputs: A dictionary that maps string keys to a tensor values.

        Returns:
            A [batch_size, sequence_length, hidden_size] float tensor.
        """
        return self.model(**inputs).logits

    def get_hidden_size(self) -> int:
        """Returns the dimension size of the output tensor.

        Returns:
            The dimension size of the output tensor.
        """
        return self.model.num_labels


class TransformerModelWithHeadEncoderFactory(BaseEncoderFactory):
    """Factory class for creating TransformerModelWithHeadEncoder instances.

    Args:
        model_name: A name or path of the pre-trained transformer model.
        dropout_prob: Dropout probability to apply. Defaults to 0.2.
    """

    def __init__(self, model_name: str, dropout_prob: float = 0.2):
        self.__model_name = model_name
        self.__dropout_prob = dropout_prob

    def create(self, label_set: LabelSet) -> TransformerModelWithHeadEncoder:
        """Creates an TransformerModelWithHeadEncoder instance based on
        the provided label set.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            An encoder that transforms input into a tensor representation.
        """
        model = AutoModelForTokenClassification.from_pretrained(
            self.__model_name,
            num_labels=label_set.get_tag_size(),
            hidden_dropout_prob=self.__dropout_prob,
        )
        return TransformerModelWithHeadEncoder(model)
