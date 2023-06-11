from __future__ import annotations

import pytest

from partial_tagger.data.core import CharBasedTags, Span, Tag
from partial_tagger.metric import Metric


@pytest.mark.parametrize(
    "predictions, ground_truths, expected",
    [
        (
            (CharBasedTags((Tag(Span(0, 5), "LOC"),), "Tokyo"),),
            (CharBasedTags((Tag(Span(0, 5), "LOC"),), "Tokyo"),),
            {"f1_score": 1.0, "precision": 1.0, "recall": 1.0},
        )
    ],
)
def test_metrics_are_valid(
    predictions: tuple[CharBasedTags, ...],
    ground_truths: tuple[CharBasedTags, ...],
    expected: dict[str, float],
) -> None:
    metric = Metric()
    metric(predictions, ground_truths)

    assert metric.get_scores() == expected
