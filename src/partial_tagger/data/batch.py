from __future__ import annotations

from abc import ABCMeta, abstractmethod
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from ..crf.functional import to_tag_bitmap
from . import CharBasedTags, LabelSet, Span, SubwordBasedTags, Tag, TokenizedText

# https://bugs.python.org/issue45117
# type alias for tuple, dict, list is no longer supported in py38.
Texts = Tuple[str, ...]
TokenizedTexts = Tuple[TokenizedText, ...]

CharBasedTagsCollection = Tuple[CharBasedTags, ...]
SubwordBasedTagsCollection = Tuple[SubwordBasedTags, ...]

Dataset = List[Tuple[str, CharBasedTags]]

TaggerInputs = Dict[str, torch.Tensor]


def pad(batch: list[list[int]], fill_value: int) -> torch.Tensor:
    max_length = max(map(len, batch))
    return torch.tensor([x + [fill_value] * (max_length - len(x)) for x in batch])


def unpad(batch: torch.Tensor, fill_value: int) -> list[list[int]]:
    return [[i for i in x if i != fill_value] for x in batch.tolist()]


class TagFactory:
    def __init__(self, tokenized_texts: TokenizedTexts, label_set: LabelSet):
        self.__tokenized_texts = tokenized_texts
        self.__label_set = label_set

    @property
    def tokenized_texts(self) -> TokenizedTexts:
        return self.__tokenized_texts

    def create_char_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> CharBasedTagsCollection:
        return tuple(
            tags.get_char_based_tags()
            for tags in self.create_subword_based_tags(tag_indices, padding_index)
        )

    def create_subword_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> SubwordBasedTagsCollection:
        label_set = self.__label_set

        batched_tags = []

        for text, indices in zip(
            self.__tokenized_texts, unpad(tag_indices, padding_index)
        ):
            tags = []
            now = 0
            for label, group in groupby(
                label_set.get_label(index) for index in indices
            ):
                length = len(list(group))

                if label is not None:
                    tags.append(Tag(Span(now, length), label))

                now += length

            batched_tags.append(SubwordBasedTags(tuple(tags), text))

        return tuple(batched_tags)

    def create_tag_indices(
        self,
        tags_collection: CharBasedTagsCollection,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        tag_indices = []
        label_set = self.__label_set

        for text, tags in zip(self.__tokenized_texts, tags_collection):
            indices = [unknown_index] * text.num_tokens

            for token_index in range(text.num_tokens):
                span = text.get_char_span(token_index)
                if span is None:
                    indices[token_index] = label_set.get_outside_index()

            for tag in tags:
                start = text.convert_to_token_index(tag.start)
                end = text.convert_to_token_index(tag.start + tag.length - 1)
                if start == end:
                    indices[start] = label_set.get_unit_index(tag.label)
                else:
                    indices[start] = label_set.get_start_index(tag.label)
                    indices[start + 1 : end] = [
                        label_set.get_inside_index(tag.label)
                    ] * (end - start - 1)
                    indices[end] = label_set.get_end_index(tag.label)

            tag_indices.append(indices)

        return pad(tag_indices, padding_index).to(device)

    def create_tag_bitmap(
        self,
        tags_collection: CharBasedTagsCollection,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        tag_indices = self.create_tag_indices(
            tags_collection, device, padding_index, unknown_index
        )
        return to_tag_bitmap(
            tag_indices, self.__label_set.get_tag_size(), unknown_index
        )


class Batch:
    def __init__(
        self, tagger_inputs: TaggerInputs, mask: torch.Tensor, tag_factory: TagFactory
    ):
        self.__tagger_inputs = tagger_inputs
        self.__mask = mask
        self.__tag_factory = tag_factory

    def get_tagger_inputs(self, device: torch.device) -> TaggerInputs:
        return {key: x.to(device) for key, x in self.__tagger_inputs.items()}

    def get_mask(self, device: torch.device) -> torch.Tensor:
        return self.__mask.to(device)

    @property
    def tokenized_texts(self) -> TokenizedTexts:
        return self.__tag_factory.tokenized_texts

    @property
    def size(self) -> int:
        return self.__mask.size(0)

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


class BaseBatchFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, texts: Texts) -> Batch:
        raise NotImplementedError


class TransformerBatchFactory(BaseBatchFactory):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_set: LabelSet,
        tokenizer_args: Optional[dict] = None,
    ):
        if not tokenizer.is_fast:
            raise ValueError("Only fast tokenizer is supported.")

        self.__tokenizer = tokenizer
        self.__label_set = label_set

        self.__tokenizer_args = tokenizer_args or {
            "padding": True,
            "return_tensors": "pt",
        }
        self.__tokenizer_args["return_offsets_mapping"] = True

    def create(self, texts: Texts) -> Batch:
        batch_encoding = self.__tokenizer(texts, **self.__tokenizer_args)

        mappings = batch_encoding.pop("offset_mapping").tolist()
        pad_token_id = self.__tokenizer.pad_token_id
        tokenized_text_lengths = (batch_encoding.input_ids != pad_token_id).sum(dim=1)

        tokenized_texts = []
        for tokenized_text_length, mapping, text in zip(
            tokenized_text_lengths, mappings, texts
        ):
            char_spans = tuple(
                Span(start, end - start) if start != end else None
                for start, end in mapping[:tokenized_text_length]
            )
            token_indices = [-1] * len(text)
            for token_index, char_span in enumerate(char_spans):
                if char_span is None:
                    continue
                start = char_span.start
                end = char_span.start + char_span.length
                token_indices[start:end] = [token_index] * char_span.length

            tokenized_texts.append(
                TokenizedText(text, char_spans, tuple(token_indices))
            )

        lengths = [text.num_tokens for text in tokenized_texts]
        max_length = max(lengths)
        mask = torch.tensor(
            [[True] * length + [False] * (max_length - length) for length in lengths]
        )
        return Batch(
            batch_encoding,
            mask,
            TagFactory(tuple(tokenized_texts), self.__label_set),
        )


class Collator:
    def __init__(self, batch_factory: BaseBatchFactory):
        self.batch_factory = batch_factory

    def __call__(self, examples: Dataset) -> tuple[Batch, CharBasedTagsCollection]:
        texts, tags_collection = zip(*examples)
        return self.batch_factory.create(texts), tags_collection
