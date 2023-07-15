from __future__ import annotations

from transformers import AutoTokenizer

from partial_tagger.data.batch.text import TransformerTokenizer
from partial_tagger.data.core import Span, Tag
from partial_tagger.encoders.transformer import (
    TransformerModelEncoderFactory,
    TransformerModelWithHeadEncoderFactory,
)
from partial_tagger.training import Trainer


def create_tag(start: int, length: int, label: str) -> Tag:
    """Creates a tag.

    Args:
        start: An integer representing a start index of a tag.
        length: An integer representing length of a tag.
        label: A string representing a label of a tag.

    Returns:
        An instance of Tag.
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
    target_entity_ratio: float = 0.15,
    entity_ratio_margin: float = 0.05,
    balancing_coefficient: int = 10,
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
    return Trainer(
        tokenizer=tokenizer,
        encoder_factory=encoder_factory,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_clip_value=gradient_clip_value,
        padding_index=padding_index,
        target_entity_ratio=target_entity_ratio,
        entity_ratio_margin=entity_ratio_margin,
        balancing_coefficient=balancing_coefficient,
    )
