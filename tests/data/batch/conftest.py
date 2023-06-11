import pytest
from transformers import AutoTokenizer

from partial_tagger.data import LabelSet
from partial_tagger.data.batch.text import TransformerTokenizer


@pytest.fixture
def tokenizer() -> TransformerTokenizer:
    return TransformerTokenizer(AutoTokenizer.from_pretrained("distilroberta-base"))


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet({"ORG", "LOC", "PER", "MISC"})
