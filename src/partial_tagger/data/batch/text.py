from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from ..core import (
    CharBasedTags,
    LabelSet,
    Span,
    Status,
    Tag,
    TokenBasedTags,
    TokenizedText,
)


def create_token_based_tags(
    tokenized_texts: tuple[TokenizedText, ...],
    tag_indices: torch.Tensor,
    label_set: LabelSet,
    padding_index: int,
) -> tuple[TokenBasedTags, ...]:
    tag_indices_unpadded = tuple(
        tuple(i for i in x if i != padding_index) for x in tag_indices.tolist()
    )

    if len(tag_indices_unpadded) != len(tokenized_texts):
        raise ValueError("Batch size mismatch.")

    tags_batch = []

    for text, indices in zip(tokenized_texts, tag_indices_unpadded):
        if text.num_tokens != len(indices):
            raise ValueError("The number of tokens in text mismatch.")

        tags = []
        stack: list[str] = []
        for pos, index in enumerate(indices):
            status = label_set.get_status(index)
            label = label_set.get_label(index)
            if status is None or label is None:
                continue

            if status == Status.UNIT:
                tags.append(Tag(Span(pos, 1), label))
            elif status == Status.END:
                if stack[-1] == label:
                    length = len(stack)
                    tags.append(Tag(Span(pos - length, length + 1), label))
                stack.clear()
            elif status == Status.START or status == Status.INSIDE:
                if not stack or stack[-1] == label:
                    stack.append(label)
                else:
                    stack.clear()
            else:
                raise ValueError("Invalid status.")

        tags_batch.append(TokenBasedTags(tuple(tags), text))

    return tuple(tags_batch)


class TextBatch:
    def __init__(
        self,
        tokenized_texts: tuple[TokenizedText, ...],
        tagger_inputs: dict[str, torch.Tensor],
        mask: torch.Tensor,
        device: torch.device | None = None,
    ):
        self.tokenized_texts = tokenized_texts
        self.__tagger_inputs = tagger_inputs
        self.__mask = mask
        self.__device = device

    @property
    def size(self) -> int:
        return len(self.tokenized_texts)

    def to(self, device: torch.device) -> None:
        self.__device = device

    @property
    def tagger_inputs(self) -> dict[str, torch.Tensor]:
        if self.__device is not None:
            return {key: x.to(self.__device) for key, x in self.__tagger_inputs.items()}
        else:
            return self.__tagger_inputs

    @property
    def mask(self) -> torch.Tensor:
        if self.__device is not None:
            return self.__mask.to(self.__device)
        else:
            return self.__mask

    def create_char_based_tags(
        self, tag_indices: torch.Tensor, label_set: LabelSet, padding_index: int = -1
    ) -> tuple[CharBasedTags, ...]:
        return tuple(
            tags.convert_to_char_based()
            for tags in self.create_token_based_tags(
                tag_indices, label_set, padding_index
            )
        )

    def create_token_based_tags(
        self, tag_indices: torch.Tensor, label_set: LabelSet, padding_index: int = -1
    ) -> tuple[TokenBasedTags, ...]:
        return create_token_based_tags(
            self.tokenized_texts, tag_indices, label_set, padding_index
        )


class BaseTokenizer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, texts: tuple[str, ...]) -> TextBatch:
        raise NotImplementedError


class TransformerTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tokenizer_args: dict[str, Any] | None = None,
    ):
        if not tokenizer.is_fast:
            raise ValueError("Only transformers.PreTrainedTokenizerFast is supported.")

        self.__tokenizer = tokenizer
        self.__tokenizer_args = tokenizer_args or {
            "padding": True,
            "return_tensors": "pt",
            "return_offsets_mapping": True,
        }

        if not self.__tokenizer_args.get("return_offsets_mapping", False):
            raise ValueError("Set return_offsets_mapping to True")

        if self.__tokenizer_args.get("return_tensors", "") != "pt":
            raise ValueError("Set return_tensors to pt")

    def __call__(self, texts: tuple[str, ...]) -> TextBatch:
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
        return TextBatch(tuple(tokenized_texts), batch_encoding, mask)
