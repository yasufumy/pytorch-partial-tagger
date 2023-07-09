from __future__ import annotations


class Metric:
    def __init__(self) -> None:
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0

    def __call__(
        self,
        predictions: tuple[Tags, ...],
        ground_truths: tuple[Tags, ...],
    ) -> None:
        for tags1, tags2 in zip(predictions, ground_truths):
            tag_set1 = {(tag1.start, tag1.length, tag1.label) for tag1 in tags1}
            tag_set2 = {(tag2.start, tag2.length, tag2.label) for tag2 in tags2}

            self.__tp += len(tag_set1 & tag_set2)
            self.__fp += len(tag_set1 - tag_set2)
            self.__fn += len(tag_set2 - tag_set1)

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
