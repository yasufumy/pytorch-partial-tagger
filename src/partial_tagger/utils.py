from __future__ import annotations

from transformers import AutoTokenizer

from .data.batch.text import TransformerTokenizer
from .data.core import Span, Tag
from .encoders.transformer import (
    TransformerModelEncoderFactory,
    TransformerModelWithHeadEncoderFactory,
)
from .training import Trainer


def create_tag(start: int, length: int, label: str) -> Tag:
    """Creates a tag.

    Args:
        start: An integer representing a start index of a tag.
        length: An integer representing length of a tag.
        label: A string representing a label of a tag.

    Returns:
        A Tag.
    """
    return Tag(Span(start, length), label)


def create_trainer(
    model_name: str = "roberta-base",
    dropout: float = 0.2,
    batch_size: int = 15,
    num_epochs: int = 20,
    learning_rate: float = 2e-5,
    gradient_clip_value: float = 5.0,
    padding_index: int = -1,
    tokenizer_args: dict | None = None,
    encoder_type: str = "default",
) -> Trainer:
    """Creates Trainer."""

    if encoder_type == "default":
        encoder_factory = TransformerModelEncoderFactory(model_name, dropout)
    elif encoder_type == "with_head":
        encoder_factory = TransformerModelWithHeadEncoderFactory(model_name, dropout)
    else:
        raise ValueError(f"{encoder_type} is not supported.")

    tokenizer = TransformerTokenizer(
        AutoTokenizer.from_pretrained(model_name), tokenizer_args
    )
    return Trainer(
        tokenizer,
        encoder_factory,
        batch_size,
        num_epochs,
        learning_rate,
        gradient_clip_value,
        padding_index,
    )
