import pytest
import torch
from transformers import AutoTokenizer

from partial_tagger.data import CharBasedTags, LabelSet, Span, Tag
from partial_tagger.data.batch import TransformerBatchFactory


@pytest.fixture
def batch_factory() -> TransformerBatchFactory:
    return TransformerBatchFactory(
        AutoTokenizer.from_pretrained("distilroberta-base"),
        LabelSet({"LOC", "MISC", "ORG", "PER"}),
    )


def test_char_based_tags_are_valid(batch_factory: TransformerBatchFactory) -> None:
    text = "Tokyo is the capital of Japan."
    tag_indices = torch.tensor([[0, 1, 3, 0, 0, 0, 0, 4, 0, 0]])

    expected = CharBasedTags(
        (Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text=text
    )

    batch = batch_factory.create((text,))
    char_based_tags_collection = batch.create_char_based_tags(tag_indices)

    assert len(char_based_tags_collection) == 1
    assert char_based_tags_collection[0] == expected


def test_tag_indices_are_valid(batch_factory: TransformerBatchFactory) -> None:
    text = "Tokyo is the capital of Japan."
    tags = CharBasedTags((Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text=text)
    unknown_index = -100

    expected = torch.tensor([[0, 1, 3, -100, -100, -100, -100, 4, -100, 0]])

    batch = batch_factory.create((text,))
    tag_indices = batch.create_tag_indices(
        (tags,), torch.device("cpu"), unknown_index=unknown_index
    )

    assert torch.equal(tag_indices, expected)
