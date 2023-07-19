from __future__ import annotations

from transformers import AutoTokenizer

from partial_tagger.data.batch.text import TransformerTokenizer
from partial_tagger.encoders.transformer import (
    TransformerModelEncoderFactory,
    TransformerModelWithHeadEncoderFactory,
)
from partial_tagger.training import Trainer


def create_trainer(
    model_name: str = "roberta-base",
    dropout: float = 0.2,
    tokenizer_args: dict | None = None,
    encoder_type: str = "default",
) -> Trainer:
    """Creates an instance of Trainer."""

    if encoder_type == "default":
        encoder_factory = TransformerModelEncoderFactory(model_name, dropout)
    elif encoder_type == "with_head":
        encoder_factory = TransformerModelWithHeadEncoderFactory(model_name, dropout)
    else:
        raise ValueError(f"{encoder_type} is not supported.")

    tokenizer = TransformerTokenizer(
        AutoTokenizer.from_pretrained(model_name), tokenizer_args
    )
    return Trainer(tokenizer=tokenizer, encoder_factory=encoder_factory)
