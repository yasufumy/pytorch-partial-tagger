from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel

from ..data.core import LabelSet
from .base import BaseEncoder, BaseEncoderFactory


class TransformerModelEncoder(BaseEncoder):
    def __init__(
        self,
        model: PreTrainedModel,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.2,
    ):
        super(TransformerModelEncoder, self).__init__()

        self.model = model
        self.linear = nn.Linear(embedding_size, hidden_size)
        self.__hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.model(**inputs).last_hidden_state
        return self.linear(self.dropout(embeddings))

    def get_hidden_size(self) -> int:
        return self.__hidden_size


class TransformerModelEncoderFactory(BaseEncoderFactory):
    def __init__(self, model_name: str, dropout_prob: float = 0.2):
        self.__model_name = model_name
        self.__dropout_prob = dropout_prob

    def create(self, label_set: LabelSet) -> TransformerModelEncoder:
        model = AutoModel.from_pretrained(self.__model_name)
        return TransformerModelEncoder(
            model,
            model.config.hidden_size,
            label_set.get_tag_size(),
            self.__dropout_prob,
        )


class TransformerModelWithHeadEncoder(BaseEncoder):
    def __init__(self, model: AutoModelForTokenClassification):
        super(TransformerModelWithHeadEncoder, self).__init__()

        self.model = model

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).logits

    def get_hidden_size(self) -> int:
        return self.model.num_labels


class TransformerModelWithHeadEncoderFactory(BaseEncoderFactory):
    def __init__(self, model_name: str, dropout_prob: float = 0.2):
        self.__model_name = model_name
        self.__dropout_prob = dropout_prob

    def create(self, label_set: LabelSet) -> TransformerModelWithHeadEncoder:
        model = AutoModelForTokenClassification.from_pretrained(
            self.__model_name,
            num_labels=label_set.get_tag_size(),
            hidden_dropout_prob=self.__dropout_prob,
        )
        return TransformerModelWithHeadEncoder(model)
