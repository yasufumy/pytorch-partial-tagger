from __future__ import annotations

import pytest

from partial_tagger.data.core import Span, Tag
from partial_tagger.metric import Metric


@pytest.mark.parametrize(
    "predictions, ground_truths, expected",
    [
        (
            ({Tag(span=Span(start=0, length=5), label="LOC")},),
            ({Tag(span=Span(start=0, length=5), label="LOC")},),
            {"f1_score": 1.0, "precision": 1.0, "recall": 1.0},
        )
    ],
)
def test_metrics_are_valid(
    predictions: tuple[set[Tag], ...],
    ground_truths: tuple[set[Tag], ...],
    expected: dict[str, float],
) -> None:
    metric = Metric()
    metric(predictions, ground_truths)

    assert metric.get_scores() == expected
