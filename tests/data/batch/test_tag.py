from __future__ import annotations

import pytest
import torch

from partial_tagger.data import CharBasedTags, LabelSet, Span, Tag
from partial_tagger.data.batch.tag import TagsBatch
from partial_tagger.data.batch.text import TransformerTokenizer


def test_tag_indices_are_valid(
    tokenizer: TransformerTokenizer, label_set: LabelSet
) -> None:
    text = "Tokyo is the capital of Japan."
    char_based_tags = CharBasedTags(
        (Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text=text
    )

    expected = torch.tensor([[0, 1, 3, -100, -100, -100, -100, 4, -100, 0]])

    text_batch = tokenizer((text,))
    tags_batch = TagsBatch(
        (char_based_tags.convert_to_token_based(text_batch.tokenized_texts[0]),),
        label_set,
    )
    tag_indices = tags_batch.get_tag_indices()

    assert torch.equal(tag_indices, expected)


params = [
    (
        "The Tokyo Metropolitan Government is the government of the Tokyo Metropolis.",
        (Tag(Span(4, 29), "ORG"), Tag(Span(4, 5), "LOC"), Tag(Span(59, 5), "LOC")),
        torch.tensor(
            [
                [
                    [
                        True,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MISC
                        False,  # I-MISC
                        False,  # L-MISC
                        False,  # U-MISC
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                    [  # The
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MISC
                        True,  # I-MISC
                        True,  # L-MISC
                        True,  # U-MISC
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [  # Tokyo
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        True,  # U-LOC
                        False,  # B-MISC
                        False,  # I-MISC
                        False,  # L-MISC
                        False,  # U-MISC
                        True,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                    [  # Metropolitan
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        True,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                    [  # Government
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        True,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                    [  # is
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # the
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # government
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # of
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # the
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # Tokyo
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        True,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,
                    ],
                    [  # Met
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # ropolis
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [  # .
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,
                    ],
                    [
                        True,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,
                    ],
                ]
            ]
        ),
    ),
    (
        "John Doe is a multiple-use placeholder name.",
        (Tag(Span(0, 4), "PER"), Tag(Span(5, 3), "PER"), Tag(Span(0, 8), "PER")),
        torch.Tensor(
            [
                [
                    [
                        True,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                    [  # John
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        True,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        True,  # U-PER
                    ],
                    [  # Doe
                        False,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        True,  # B-LOC
                        True,  # I-LOC
                        True,  # L-LOC
                        True,  # U-LOC
                        True,  # B-MIS
                        True,  # I-MIS
                        True,  # L-MIS
                        True,  # U-MIS
                        True,  # B-ORG
                        True,  # I-ORG
                        True,  # L-ORG
                        True,  # U-ORG
                        True,  # B-PER
                        True,  # I-PER
                        True,  # L-PER
                        True,  # U-PER
                    ],
                    [
                        True,  # O
                        False,  # B-LOC
                        False,  # I-LOC
                        False,  # L-LOC
                        False,  # U-LOC
                        False,  # B-MIS
                        False,  # I-MIS
                        False,  # L-MIS
                        False,  # U-MIS
                        False,  # B-ORG
                        False,  # I-ORG
                        False,  # L-ORG
                        False,  # U-ORG
                        False,  # B-PER
                        False,  # I-PER
                        False,  # L-PER
                        False,  # U-PER
                    ],
                ]
            ]
        ),
    ),
]


@pytest.mark.parametrize("text, tags, expected", params)
def test_tag_bitmap_is_valid(
    label_set: LabelSet,
    tokenizer: TransformerTokenizer,
    text: str,
    tags: tuple[Tag],
    expected: torch.Tensor,
) -> None:
    char_based_tags = CharBasedTags(
        tags,
        text=text,
    )

    text_batch = tokenizer((text,))
    tags_batch = TagsBatch(
        (char_based_tags.convert_to_token_based(text_batch.tokenized_texts[0]),),
        label_set,
    )

    tag_bitmap = tags_batch.get_tag_bitmap()

    assert torch.equal(tag_bitmap, expected)
