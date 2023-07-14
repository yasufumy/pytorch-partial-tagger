from __future__ import annotations

from partial_tagger.data.batch.tag import TagsBatch
from partial_tagger.data.batch.text import BaseTokenizer, TextBatch
from partial_tagger.data.core import LabelSet, Tag


class Collator:
    def __init__(self, tokenizer: BaseTokenizer, label_set: LabelSet):
        self.__tokenizer = tokenizer
        self.__label_set = label_set

    def __call__(
        self, examples: list[tuple[str, set[Tag]]]
    ) -> tuple[TextBatch, TagsBatch]:
        texts, tags_batch = zip(*examples)

        text_batch = self.__tokenizer(texts=texts)

        tags_batch = TagsBatch(
            tags_batch=tags_batch,
            label_set=self.__label_set,
            alignments=text_batch.alignments,
        )
        return text_batch, tags_batch
