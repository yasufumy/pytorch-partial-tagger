from typing import Literal

import torch
from torch import nn
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel

from .base import BaseEncoder, TaggerInputs


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

    def forward(self, inputs: TaggerInputs) -> torch.Tensor:
        embeddings = self.model(**inputs).last_hidden_state
        return self.linear(self.dropout(embeddings))

    def get_hidden_size(self) -> int:
        return self.__hidden_size


class TransformerModelWithHeadEncoder(BaseEncoder):
    def __init__(self, model: AutoModelForTokenClassification):
        super(TransformerModelWithHeadEncoder, self).__init__()

        self.model = model

    def forward(self, inputs: TaggerInputs) -> torch.Tensor:
        return self.model(**inputs).logits

    def get_hidden_size(self) -> int:
        return self.model.num_labels


EncoderType = Literal["default", "with_head"]


def create_encoder(
    encoder_type: EncoderType, model_name: str, num_tags: int, dropout_prob: float = 0.2
) -> BaseEncoder:
    if encoder_type == "default":
        model = AutoModel.from_pretrained(model_name)
        return TransformerModelEncoder(
            model, model.config.hidden_size, num_tags, dropout_prob
        )
    elif encoder_type == "with_head":
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_tags, hidden_dropout_prob=dropout_prob
        )
        return TransformerModelWithHeadEncoder(model)
    else:
        raise ValueError("A specified encoder is not supported.")
