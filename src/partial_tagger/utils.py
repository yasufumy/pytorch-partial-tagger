from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from transformers import AutoTokenizer

from partial_tagger.data.collators import TransformerCollator
from partial_tagger.data.core import Span, Tag
from partial_tagger.encoders.transformer import (
    TransformerModelEncoderFactory,
    TransformerModelWithHeadEncoderFactory,
)
from partial_tagger.training import Trainer

if TYPE_CHECKING:
    from partial_tagger.encoders.base import BaseEncoderFactory


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
    tokenizer_args: dict[str, Any] | None = None,
    encoder_type: Literal["default", "with_head"] = "default",
) -> Trainer:
    """Creates an instance of Trainer."""

    encoder_factory: BaseEncoderFactory

    if encoder_type == "default":
        encoder_factory = TransformerModelEncoderFactory(model_name, dropout)
    elif encoder_type == "with_head":
        encoder_factory = TransformerModelWithHeadEncoderFactory(model_name, dropout)
    else:
        raise ValueError(f"{encoder_type} is not supported.")

    collator = TransformerCollator(
        AutoTokenizer.from_pretrained(model_name), tokenizer_args
    )
    return Trainer(collator=collator, encoder_factory=encoder_factory)
