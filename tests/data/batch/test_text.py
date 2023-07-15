from __future__ import annotations

import pytest
import torch

from partial_tagger.data import LabelSet, Span, Tag
from partial_tagger.data.batch.text import TransformerTokenizer


@pytest.mark.parametrize(
    "text, tag_indices, tags",
    [
        (
            "Tokyo is the capital of Japan.",
            torch.tensor([[0, 1, 3, 0, 0, 0, 0, 4, 0, 0]]),
            {Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")},
        ),
        (
            "John Doe",
            torch.tensor([[0, 16, 16, 0]]),
            {Tag(Span(0, 4), "PER"), Tag(Span(5, 3), "PER")},
        ),
        ("John Doe", torch.tensor([[0, 13, 15, 0]]), {Tag(Span(0, 8), "PER")}),
    ],
)
def test_char_based_tags_are_valid(
    tokenizer: TransformerTokenizer,
    label_set: LabelSet,
    text: str,
    tag_indices: torch.Tensor,
    tags: set[Tag],
) -> None:
    text_batch = tokenizer((text,))

    char_based_tags_batch = text_batch.create_char_based_tags(tag_indices, label_set)

    assert len(char_based_tags_batch) == 1
    assert char_based_tags_batch[0] == tags
