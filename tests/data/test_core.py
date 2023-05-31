import pytest

from partial_tagger.data import LabelSet, Span, TokenizedText


@pytest.fixture
def tokenized_text() -> TokenizedText:
    # Tokenized by RoBERTa
    return TokenizedText(
        "Tokyo is the capital of Japan.",
        (
            None,
            Span(0, 3),
            Span(3, 2),
            Span(6, 2),
            Span(9, 3),
            Span(13, 7),
            Span(21, 2),
            Span(24, 5),
            Span(29, 1),
            None,
        ),
        (
            1,
            1,
            1,
            2,
            2,
            -1,
            3,
            3,
            -1,
            4,
            4,
            4,
            -1,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            -1,
            6,
            6,
            -1,
            7,
            7,
            7,
            7,
            7,
            8,
        ),
    )


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet({"ORG", "PER"})


def test_tokens_are_valid(tokenized_text: TokenizedText) -> None:
    token_indices = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = ["Tok", "yo", "is", "the", "capital", "of", "Japan", "."]

    tokens = []
    for token_index in token_indices:
        tokens.append(tokenized_text.get_token(token_index))

    assert tokens == expected


def test_converts_token_span_to_char_span(tokenized_text: TokenizedText) -> None:
    token_spans = [
        Span(1, 2),  # Tok yo
        Span(3, 1),  # is
        Span(4, 1),  # the
        Span(5, 1),  # capital
        Span(6, 1),  # of
        Span(7, 1),  # Japan
        Span(8, 1),  # .
    ]
    expected = [
        Span(0, 5),  # Tokyo
        Span(6, 2),  # is
        Span(9, 3),  # the
        Span(13, 7),  # capital
        Span(21, 2),  # of
        Span(24, 5),  # Japan
        Span(29, 1),  # .
    ]

    char_spans = []
    for token_span in token_spans:
        char_spans.append(tokenized_text.convert_to_char_span(token_span))

    assert char_spans == expected


def test_label_is_valid(label_set: LabelSet) -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    expected = [None] + ["ORG"] * 4 + ["PER"] * 4

    labels = [label_set.get_label(index) for index in indices]

    assert labels == expected


def test_start_states_are_valid(label_set: LabelSet) -> None:
    expected = [
        True,  # O
        True,  # B-ORG
        False,  # I-ORG
        False,  # L-ORG
        True,  # U-ORG
        True,  # B-PER
        False,  # I-PER
        False,  # L-PER
        True,  # U-PER
    ]

    assert label_set.get_start_states() == expected


def test_end_states_are_valid(label_set: LabelSet) -> None:
    expected = [
        True,  # O
        False,  # B-ORG
        False,  # I-ORG
        True,  # L-ORG
        True,  # U-ORG
        False,  # B-PER
        False,  # I-PER
        True,  # L-PER
        True,  # U-PER
    ]

    assert label_set.get_end_states() == expected


def test_transitions_are_valid(label_set: LabelSet) -> None:
    expected = [
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # O
        [
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ],  # B-ORG
        [
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ],  # I-ORG
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # L-ORG
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # U-ORG
        [
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ],  # B-PER
        [
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ],  # I-PER
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # L-PER
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # U-PER
    ]

    assert label_set.get_transitions() == expected
