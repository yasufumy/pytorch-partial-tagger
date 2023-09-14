from __future__ import annotations

import pytest
from sequence_label import SequenceLabel

from partial_tagger.metric import Metric


@pytest.mark.parametrize(
    ("predictions", "ground_truths", "expected"),
    [
        (
            (
                SequenceLabel.from_dict(
                    [{"start": 0, "end": 5, "label": "LOC"}], size=10
                ),
            ),
            (
                SequenceLabel.from_dict(
                    [{"start": 0, "end": 5, "label": "LOC"}], size=10
                ),
            ),
            {"f1_score": 1.0, "precision": 1.0, "recall": 1.0},
        )
    ],
)
def test_metrics_are_valid(
    predictions: tuple[SequenceLabel, ...],
    ground_truths: tuple[SequenceLabel, ...],
    expected: dict[str, float],
) -> None:
    metric = Metric()
    metric(predictions, ground_truths)

    assert metric.get_scores() == expected
