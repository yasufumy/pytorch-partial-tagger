from __future__ import annotations

from itertools import groupby
from typing import Tuple

import torch

from ...crf.functional import to_tag_bitmap
from .. import CharBasedTags, LabelSet, Span, Tag, TokenBasedTags, TokenizedText

# https://bugs.python.org/issue45117
# type alias for tuple, dict, list is no longer supported in py38.
CharBasedTagsBatch = Tuple[CharBasedTags, ...]
TokenBasedTagsBatch = Tuple[TokenBasedTags, ...]


def pad(batch: list[list[int]], fill_value: int) -> torch.Tensor:
    max_length = max(map(len, batch))
    return torch.tensor([x + [fill_value] * (max_length - len(x)) for x in batch])


def unpad(batch: torch.Tensor, fill_value: int) -> list[list[int]]:
    return [[i for i in x if i != fill_value] for x in batch.tolist()]


class TagFactory:
    def __init__(self, tokenized_texts: tuple[TokenizedText, ...], label_set: LabelSet):
        self.__tokenized_texts = tokenized_texts
        self.__label_set = label_set

    def create_char_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> CharBasedTagsBatch:
        return tuple(
            tags.get_char_based_tags()
            for tags in self.create_token_based_tags(tag_indices, padding_index)
        )

    def create_token_based_tags(
        self, tag_indices: torch.Tensor, padding_index: int = -1
    ) -> TokenBasedTagsBatch:
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

            batched_tags.append(TokenBasedTags(tuple(tags), text))

        return tuple(batched_tags)

    def create_tag_indices(
        self,
        tags_batch: CharBasedTagsBatch,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        tag_indices = []
        label_set = self.__label_set

        for text, tags in zip(self.__tokenized_texts, tags_batch):
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
        tags_batch: CharBasedTagsBatch,
        device: torch.device,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> torch.Tensor:
        tag_indices = self.create_tag_indices(
            tags_batch, device, padding_index, unknown_index
        )
        return to_tag_bitmap(
            tag_indices, self.__label_set.get_tag_size(), unknown_index
        )
