import random

import numpy as np
import torch
from sequence_label import SequenceLabel

from partial_tagger.utils import create_trainer


def test_recognizer_outputs_valid_char_based_tags() -> None:
    text = "Tokyo is the capital of Japan."
    label = SequenceLabel.from_dict(
        tags=[
            {"start": 0, "end": 5, "label": "LOC"},
            {"start": 24, "end": 29, "label": "LOC"},
        ],
        size=len(text),
    )
    expected = (
        SequenceLabel.from_dict(
            tags=[{"start": 24, "end": 29, "label": "LOC"}], size=len(text)
        ),
    )
    dataset = [(text, label)]
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
