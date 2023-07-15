from __future__ import annotations

from partial_tagger.data import Tag


class Metric:
    def __init__(self) -> None:
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0

    def __call__(
        self,
        predictions: tuple[set[Tag], ...],
        ground_truths: tuple[set[Tag], ...],
    ) -> None:
        for tags1, tags2 in zip(predictions, ground_truths):
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
