import random

import numpy as np
import torch

from partial_tagger.data import Span, Tag
from partial_tagger.utils import create_trainer


def test_recognizer_outputs_valid_char_based_tags() -> None:
    text = "Tokyo is the capital of Japan."
    tags = {
        Tag(span=Span(start=0, length=5), label="LOC"),
        Tag(span=Span(start=24, length=5), label="LOC"),
    }
    expected = ({Tag(span=Span(start=24, length=5), label="LOC")},)
    dataset = [(text, tags)]
    device = torch.device("cpu")

    # fix seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    trainer = create_trainer(model_name="distilroberta-base")
    recognizer = trainer(dataset, dataset, device)

    output = recognizer((text,), 1, device)

    assert len(output) == 1
    assert output == expected
