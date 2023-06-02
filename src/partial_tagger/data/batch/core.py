from __future__ import annotations

from typing import List, Tuple

import torch

from .. import CharBasedTags, LabelSet, TokenizedText
from .tag import CharBasedTagsCollection, SubwordBasedTagsCollection, TagFactory
from .text import BaseTokenizer, TaggerInputs, Texts, TokenizedTexts

Dataset = List[Tuple[str, CharBasedTags]]


class Batch:
    def __init__(self, tokenized_texts: TokenizedTexts, tag_factory: TagFactory):
        self.__tokenized_texts = tokenized_texts
        self.__tag_factory = tag_factory

    def get_tagger_inputs(self, device: torch.device) -> TaggerInputs:
        return self.__tokenized_texts.get_tagger_inputs(device)

    def get_mask(self, device: torch.device) -> torch.Tensor:
        return self.__tokenized_texts.get_mask(device)

    @property
    def tokenized_texts(self) -> tuple[TokenizedText, ...]:
        return self.__tokenized_texts.tokenized_texts

    @property
    def size(self) -> int:
        return self.__tokenized_texts.size

    def create_char_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> CharBasedTagsCollection:
        return self.__tag_factory.create_char_based_tags(tag_indices, padding_index)

    def create_subword_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> SubwordBasedTagsCollection:
        return self.__tag_factory.create_subword_based_tags(tag_indices, padding_index)

    def create_tag_indices(
        self,
        tags_collection: CharBasedTagsCollection,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        return self.__tag_factory.create_tag_indices(
            tags_collection, device, padding_index, unknown_index
        )

    def create_tag_bitmap(
        self,
        tags_collection: CharBasedTagsCollection,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        return self.__tag_factory.create_tag_bitmap(
            tags_collection, device, padding_index, unknown_index
        )

    def __repr__(self) -> str:
        text_str = []
        for text in self.tokenized_texts:
            text_str.append(f"    - {repr(text)}\n")
        return "Batch:\n" + f"  size: {self.size}\n" + "  data:\n" + "".join(text_str)


class BatchFactory:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        label_set: LabelSet,
    ):
        self.__tokenizer = tokenizer
        self.__label_set = label_set

    def create(self, texts: Texts) -> Batch:
        tokenized_texts = self.__tokenizer(texts)
        tag_factory = TagFactory(tokenized_texts.tokenized_texts, self.__label_set)
        return Batch(tokenized_texts, tag_factory)


class Collator:
    def __init__(self, batch_factory: BatchFactory):
        self.batch_factory = batch_factory

    def __call__(self, examples: Dataset) -> tuple[Batch, CharBasedTagsCollection]:
        texts, tags_collection = zip(*examples)
        return self.batch_factory.create(texts), tags_collection
