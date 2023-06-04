import pytest
import torch
from transformers import AutoTokenizer

from partial_tagger.data import CharBasedTags, LabelSet, Span, Tag
from partial_tagger.data.batch.tag import TagFactory
from partial_tagger.data.batch.text import TransformerTokenizer


@pytest.fixture
def tokenizer() -> TransformerTokenizer:
    return TransformerTokenizer(AutoTokenizer.from_pretrained("distilroberta-base"))


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet({"ORG", "LOC", "PER", "MISC"})


def test_char_based_tags_are_valid(
    tokenizer: TransformerTokenizer, label_set: LabelSet
) -> None:
    text = "Tokyo is the capital of Japan."
    tag_indices = torch.tensor([[0, 1, 3, 0, 0, 0, 0, 4, 0, 0]])

    expected = CharBasedTags(
        (Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text=text
    )

    text_batch = tokenizer((text,))
    tag_factory = TagFactory(text_batch.tokenized_texts, label_set)

    char_based_tags_batch = tag_factory.create_char_based_tags(tag_indices)

    assert len(char_based_tags_batch) == 1
    assert char_based_tags_batch[0] == expected


def test_tag_indices_are_valid(
    tokenizer: TransformerTokenizer, label_set: LabelSet
) -> None:
    text = "Tokyo is the capital of Japan."
    tags = CharBasedTags((Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text=text)
    unknown_index = -100

    expected = torch.tensor([[0, 1, 3, -100, -100, -100, -100, 4, -100, 0]])

    text_batch = tokenizer((text,))
    tag_factory = TagFactory(text_batch.tokenized_texts, label_set)

    tag_indices = tag_factory.create_tag_indices(
        (tags,), torch.device("cpu"), unknown_index=unknown_index
    )

    assert torch.equal(tag_indices, expected)
