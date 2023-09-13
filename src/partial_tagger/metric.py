from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequence_label import SequenceLabel


class Metric:
    def __init__(self) -> None:
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0

    def __call__(
        self,
        predictions: tuple[SequenceLabel, ...],
        ground_truths: tuple[SequenceLabel, ...],
    ) -> None:
        for label1, label2 in zip(predictions, ground_truths):
            tags1 = set(label1.tags)
            tags2 = set(label2.tags)
            self.__tp += len(tags1 & tags2)
            self.__fp += len(tags1 - tags2)
            self.__fn += len(tags2 - tags1)

    def get_scores(self) -> dict[str, float]:
        if self.__tp + self.__fp != 0:
            precision = self.__tp / (self.__tp + self.__fp)
        else:
            precision = 0.0

        if self.__tp + self.__fn != 0:
            recall = self.__tp / (self.__tp + self.__fn)
        else:
            recall = 0.0

        if precision + recall == 0.0:
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
        else:
            return {
                "f1_score": 2 * precision * recall / (precision + recall),
                "precision": precision,
                "recall": recall,
            }
