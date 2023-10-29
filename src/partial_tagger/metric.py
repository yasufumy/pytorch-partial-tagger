from __future__ import annotations

from collections import Counter, defaultdict
from itertools import chain
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequence_label import SequenceLabel
    from sequence_label.core import Tag


class MetricDict:
    precision: float
    recall: float
    f1_score: float


class Metric:
    def __init__(self) -> None:
        self.__tp: Counter[str] = Counter()
        self.__fp: Counter[str] = Counter()
        self.__fn: Counter[str] = Counter()

    def __call__(
        self,
        predictions: tuple[SequenceLabel, ...],
        ground_truths: tuple[SequenceLabel, ...],
    ) -> None:
        for y, t in zip(predictions, ground_truths):
            y_tags = self.__get_tags_by_label(y)
            t_tags = self.__get_tags_by_label(t)
            labels = set(chain(y_tags.keys(), t_tags.keys()))
            for label in labels:
                self.__tp[label] += len(y_tags[label] & t_tags[label])
                self.__fp[label] += len(y_tags[label] - t_tags[label])
                self.__fn[label] += len(t_tags[label] - y_tags[label])

    def get_scores(self) -> dict[str, float]:
        precisions = []
        recalls = []
        f1_scores = []
        for label in sorted(self.__tp.keys()):
            metrics = self.__compute_metrics(
                tp=self.__tp[label], fp=self.__fp[label], fn=self.__fn[label]
            )
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1_scores.append(metrics["f1_score"])

        metrics = self.__compute_metrics(
            tp=sum(self.__tp.values()),
            fp=sum(self.__fp.values()),
            fn=sum(self.__fn.values()),
        )

        return {
            "micro_precision": metrics["precision"],
            "micro_recall": metrics["recall"],
            "micro_f1_score": metrics["f1_score"],
            "macro_precision": mean(precisions),
            "macro_recall": mean(recalls),
            "macro_f1_score": mean(f1_scores),
        }

    @staticmethod
    def __get_tags_by_label(label: SequenceLabel) -> dict[str, set[Tag]]:
        tags = defaultdict(set)
        for tag in label.tags:
            tags[tag.label].add(tag)
        return tags

    @staticmethod
    def __compute_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp != 0 else 0.0
        recall = tp / (tp + fn) if tp + fn != 0 else 0.0

        if precision + recall == 0.0:
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0}
        else:
            return {
                "f1_score": 2 * precision * recall / (precision + recall),
                "precision": precision,
                "recall": recall,
            }
