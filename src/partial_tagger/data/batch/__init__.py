from __future__ import annotations

from partial_tagger.data.batch.tag import TagsBatch
from partial_tagger.data.batch.text import BaseTokenizer, TextBatch
from partial_tagger.data.core import LabelSet


class Collator:
    def __init__(self, tokenizer: BaseTokenizer, label_set: LabelSet):
        self.__tokenizer = tokenizer
        self.__label_set = label_set

    def __call__(self, examples: list[tuple[str, Tags]]) -> tuple[TextBatch, TagsBatch]:
        texts: tuple[str, ...]
        char_based_tags_batch: tuple[Tags, ...]

        texts, char_based_tags_batch = zip(*examples)

        text_batch = self.__tokenizer(texts)

        tags_batch = TagsBatch(
            tuple(
                tags.change_alignment(alignment=alignment)
                for tags, alignment in zip(char_based_tags_batch, text_batch.alignments)
            ),
            self.__label_set,
        )
        return text_batch, tags_batch
