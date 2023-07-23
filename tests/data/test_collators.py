from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from partial_tagger.data import LabelSet, Span, Tag
from partial_tagger.data.collators import TransformerCollator


@pytest.fixture
def collator() -> TransformerCollator:
    return TransformerCollator(
        AutoTokenizer.from_pretrained("distilroberta-base"),
        {
            "padding": True,
            "return_tensors": "pt",
            "return_offsets_mapping": True,
            "truncation": True,
        },
    )


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet({"ORG", "LOC", "PER", "MISC"})


@pytest.mark.parametrize(
    "text, tag_indices, tags",
    [
        (
            "Tokyo is the capital of Japan.",
            [[0, 1, 3, 0, 0, 0, 0, 4, 0, 0]],
            {Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")},
        ),
        (
            "John Doe",
            [[0, 16, 16, 0]],
            {Tag(Span(0, 4), "PER"), Tag(Span(5, 3), "PER")},
        ),
        ("John Doe", [[0, 13, 15, 0]], {Tag(Span(0, 8), "PER")}),
    ],
)
def test_char_based_tags_are_valid(
    collator: TransformerCollator,
    label_set: LabelSet,
    text: str,
    tag_indices: list[list[int]],
    tags: set[Tag],
) -> None:
    _, alignments = collator((text,))

    char_based_tags_batch = alignments.create_char_based_tags(tag_indices, label_set)

    assert len(char_based_tags_batch) == 1
    assert char_based_tags_batch[0] == tags


@pytest.mark.parametrize(
    "text, char_based_tags, expected",
    [
        (
            "Tokyo is the capital of Japan.",
            {Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")},
            [[0, 1, 3, -100, -100, -100, -100, 4, -100, 0]],
        ),
        (
            "Tokyo is the capital of Japan." * 100,
            {Tag(Span(0 + 30 * i, 5), "LOC") for i in range(100)}
            | {Tag(Span(24 + 30 * i, 5), "LOC") for i in range(100)},
            [
                [0]
                + [1, 3, -100, -100, -100, -100, 4, -100] * 63
                + [1, 3, -100, -100, -100, -100]
                + [0]
            ],
        ),
    ],
)
def test_tag_indices_are_valid(
    collator: TransformerCollator,
    label_set: LabelSet,
    text: str,
    char_based_tags: set[Tag],
    expected: list[list[int]],
) -> None:
    batch, alignments = collator((text,))
    tag_indices = alignments.get_tag_indices(
        tags_batch=(char_based_tags,), label_set=label_set
    )

    assert tag_indices == expected


params = [
    (
        "The Tokyo Metropolitan Government is the government of the Tokyo Metropolis.",
        {Tag(Span(4, 29), "ORG"), Tag(Span(4, 5), "LOC"), Tag(Span(59, 5), "LOC")},
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
        ],
    ),
    (
        "John Doe is a multiple-use placeholder name.",
        {Tag(Span(0, 4), "PER"), Tag(Span(5, 3), "PER"), Tag(Span(0, 8), "PER")},
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
        ],
    ),
]


@pytest.mark.parametrize("text, tags, expected", params)
def test_tag_bitmap_is_valid(
    label_set: LabelSet,
    collator: TransformerCollator,
    text: str,
    tags: set[Tag],
    expected: list[list[list[bool]]],
) -> None:
    _, alignments = collator((text,))

    tag_bitmap = alignments.get_tag_bitmap(tags_batch=(tags,), label_set=label_set)

    assert tag_bitmap == expected
