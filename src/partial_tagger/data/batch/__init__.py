from __future__ import annotations

from partial_tagger.data.batch.tag import TagsBatch
from partial_tagger.data.batch.text import BaseTokenizer, TextBatch
from partial_tagger.data.core import CharBasedTags, LabelSet


class Collator:
    def __init__(self, tokenizer: BaseTokenizer, label_set: LabelSet):
        self.__tokenizer = tokenizer
        self.__label_set = label_set

    def __call__(
        self, examples: list[tuple[str, CharBasedTags]]
    ) -> tuple[TextBatch, TagsBatch]:
        texts, char_based_tags_batch = zip(*examples)
        text_batch = self.__tokenizer(texts)

        tags_batch = TagsBatch(
            tuple(
                tags.convert_to_token_based(text)
                for tags, text in zip(char_based_tags_batch, text_batch.tokenized_texts)
            ),
            self.__label_set,
        )
        return text_batch, tags_batch
