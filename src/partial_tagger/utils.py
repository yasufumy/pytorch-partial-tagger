from __future__ import annotations

from transformers import AutoTokenizer

from .data import Dataset, LabelSet, Span, Tag
from .data.batch.tag import CharBasedTagsBatch
from .data.batch.text import BaseTokenizer, TextBatch, TransformerTokenizer
from .decoders.viterbi import Contrainer, ViterbiDecoder
from .encoders.transformer import EncoderType, create_encoder
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
    model_name: str,
    label_set: LabelSet,
    padding_index: int,
    encoder_type: EncoderType = "default",
) -> SequenceTagger:
    """Creates a Transformer tagger.

    Args:
        model_name: A string representing a transformer's model name.
        label_set: A LabelSet object.
        padding_index: An integer representing an index to fill with.

    Returns:
        A tagger.
    """
    tagger = SequenceTagger(
        create_encoder(encoder_type, model_name, label_set.get_tag_size(), 0.2),
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


def create_tokenizer(
    model_name: str, tokenizer_args: dict | None = None
) -> TransformerTokenizer:
    """Creates a transformer tokenizer.

    Args:
        model_name: A string representing a transformer's model name.
        tokenizer_args: A dictionary representing arguments passed to tokenizer.
    """
    return TransformerTokenizer(
        AutoTokenizer.from_pretrained(model_name), tokenizer_args
    )


class Collator:
    def __init__(self, tokenizer: BaseTokenizer):
        self.__tokenizer = tokenizer

    def __call__(self, examples: Dataset) -> tuple[TextBatch, CharBasedTagsBatch]:
        texts, tags_batch = zip(*examples)
        return self.__tokenizer(texts), tags_batch


class Metric:
    def __init__(self) -> None:
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0

    def __call__(
        self,
        predictions: CharBasedTagsBatch,
        ground_truths: CharBasedTagsBatch,
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
