from __future__ import annotations

from transformers import AutoModel, AutoTokenizer

from .data import LabelSet, Span, Tag
from .data.batch import CharBasedTagsCollection, Collator, TransformerBatchFactory
from .decoders.viterbi import Contrainer, ViterbiDecoder
from .embedders import TransformerEmbedder
from .encoders.linear import LinearEncoder
from .tagger import SequenceTagger


def create_tag(start: int, length: int, label: str) -> Tag:
    """Creates a tag.

    Args:
        start: An integer representing a start index of a tag.
        length: An integer representing length of a tag.
        label: A string representing a label of a tag.

    Returns:
        A Tag.
    """
    return Tag(Span(start, length), label)


def create_tagger(
    model_name: str, label_set: LabelSet, padding_index: int
) -> SequenceTagger:
    """Creates a Transformer tagger.

    Args:
        model_name: A string representing a transformer's model name.
        label_set: A LabelSet object.
        padding_index: An integer representing an index to fill with.

    Returns:
        A tagger.
    """
    model = AutoModel.from_pretrained(model_name)
    tagger = SequenceTagger(
        TransformerEmbedder(model),
        LinearEncoder(model.config.hidden_size, label_set.get_tag_size()),
        ViterbiDecoder(
            padding_index,
            Contrainer(
                label_set.get_start_states(),
                label_set.get_end_states(),
                label_set.get_transitions(),
            ),
        ),
    )
    return tagger


def create_collator(
    model_name: str, label_set: LabelSet, tokenizer_args: dict | None = None
) -> Collator:
    """Creates a collator.

    Args:
        model_name: A string representing a transformer's model name.
        label_set: A LabelSet object.
        tokenizer_args: A dictionary representing arguments passed to tokenizer.
    """
    batch_factory = TransformerBatchFactory(
        AutoTokenizer.from_pretrained(model_name), label_set, tokenizer_args
    )
    return Collator(batch_factory)


class Metric:
    def __init__(self) -> None:
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0

    def __call__(
        self,
        predictions: CharBasedTagsCollection,
        ground_truths: CharBasedTagsCollection,
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
