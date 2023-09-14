from __future__ import annotations

import pytest
import torch
from sequence_label import LabelSet, SequenceLabel
from transformers import AutoTokenizer

from partial_tagger.data.collators import TransformerCollator
from partial_tagger.training import create_tag_bitmap


@pytest.fixture()
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


@pytest.fixture()
def label_set() -> LabelSet:
    return LabelSet({"ORG", "LOC", "PER", "MISC"})


@pytest.mark.parametrize(
    ("text", "tag_indices", "expected"),
    [
        (
            "Tokyo is the capital of Japan.",
            [[0, 1, 3, 0, 0, 0, 0, 4, 0, 0]],
            (
                SequenceLabel.from_dict(
                    [
                        {"start": 0, "end": 5, "label": "LOC"},
                        {"start": 24, "end": 29, "label": "LOC"},
                    ],
                    size=30,
                ),
            ),
        ),
        (
            "John Doe",
            [[0, 16, 16, 0]],
            (
                SequenceLabel.from_dict(
                    [
                        {"start": 0, "end": 4, "label": "PER"},
                        {"start": 5, "end": 8, "label": "PER"},
                    ],
                    size=8,
                ),
            ),
        ),
        (
            "John Doe",
            [[0, 13, 15, 0]],
            (
                SequenceLabel.from_dict(
                    [
                        {"start": 0, "end": 8, "label": "PER"},
                    ],
                    size=8,
                ),
            ),
        ),
    ],
)
def test_char_based_tags_are_valid(
    collator: TransformerCollator,
    label_set: LabelSet,
    text: str,
    tag_indices: list[list[int]],
    expected: tuple[SequenceLabel, ...],
) -> None:
    _, alignments = collator((text,))

    labels = label_set.decode(tag_indices=tag_indices, alignments=alignments)

    assert labels == expected


params = [
    (
        "The Tokyo Metropolitan Government is the government of the Tokyo Metropolis.",
        (
            SequenceLabel.from_dict(
                [
                    {"start": 4, "end": 29, "label": "ORG"},
                    {"start": 4, "end": 9, "label": "LOC"},
                    {"start": 59, "end": 64, "label": "LOC"},
                ],
                size=76,
            ),
        ),
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
        (
            SequenceLabel.from_dict(
                [
                    {"start": 0, "end": 4, "label": "PER"},
                    {"start": 5, "end": 8, "label": "PER"},
                    {"start": 0, "end": 8, "label": "PER"},
                ],
                size=44,
            ),
        ),
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


@pytest.mark.parametrize(("text", "labels", "expected"), params)
def test_tag_bitmap_is_valid(
    label_set: LabelSet,
    collator: TransformerCollator,
    text: str,
    labels: tuple[SequenceLabel, ...],
    expected: list[list[list[bool]]],
) -> None:
    _, alignments = collator((text,))

    tag_bitmap = create_tag_bitmap(
        label_set=label_set,
        labels=labels,
        alignments=alignments,
        device=torch.device("cpu"),
    ).tolist()

    assert tag_bitmap == expected
