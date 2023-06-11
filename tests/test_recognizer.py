import torch

from partial_tagger.data import CharBasedTags, Span, Tag
from partial_tagger.utils import create_trainer


def test_recognizer_outputs_valid_char_based_tags() -> None:
    text = "Tokyo is the capital of Japan."
    tags = CharBasedTags((Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")), text)
    dataset = [(text, tags)]
    device = torch.device("cpu")

    trainer = create_trainer("distilroberta-base")
    recognizer = trainer(dataset, dataset, device)

    output = recognizer((text,), 1, device)

    assert isinstance(output[0], CharBasedTags)
