from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from partial_tagger.data.core import Alignment, LabelSet, Span, Tag


class TextBatch:
    """A batch of text data for tagging.

    Args:
        tagger_inputs: A dictionary that maps string keys to a tensor values.
        mask: A [batch_size, sequence_length] float tensor representing
            a mask for a batch.
        alignments: A tuple of instances of Alignment.
        device: A device on which to place tensors. Defaults to None.

    Attributes:
        tagger_inputs: A dictionary that maps string keys to a tensor values.
        mask: A [batch_size, sequence_length] float tensor representing
            a mask for a batch.
        alignments: A tuple of instances of Alignment.
        size: An integer representing batch size.
    """

    def __init__(
        self,
        tagger_inputs: dict[str, torch.Tensor],
        mask: torch.Tensor,
        alignments: tuple[Alignment, ...],
        device: torch.device | None = None,
    ):
        self.alignments = alignments
        self.__tagger_inputs = tagger_inputs
        self.__mask = mask
        self.__device = device

    @property
    def size(self) -> int:
        return len(self.alignments)

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
    ) -> tuple[set[Tag], ...]:
        """Creates character-based tags for text batch based on given tag indices
        and an instance of LabelSet.

        Args:
            tag_indices: A [batch_size, sequence_length] integer tensor of tag indices.
            label_set: An instance of LabelSet to use for tag conversion.
            padding_index: An integer representing a padding index. Defaults to -1.

        Returns:
            A tuple where each item is a set of character-based tags.
        """
        tag_indices_unpadded = tuple(
            tuple(i for i in x if i != padding_index) for x in tag_indices.tolist()
        )

        if len(tag_indices_unpadded) != len(self.alignments):
            raise ValueError("Batch size mismatch.")

        tags_batch = []

        for alignment, indices in zip(self.alignments, tag_indices_unpadded):
            tags_batch.append(
                alignment.create_char_based_tags(
                    tag_indices=indices, label_set=label_set
                )
            )

        return tuple(tags_batch)


class BaseTokenizer(metaclass=ABCMeta):
    """Base class for all tokenizers."""

    @abstractmethod
    def __call__(self, texts: tuple[str, ...]) -> TextBatch:
        """Tokenize given texts, encode to tensors and return an instance of TextBatch.

        Args:
            texts: A tuple of strings, where each item represents a text.

        Returns:
            An instance of TextBatch.
        """
        raise NotImplementedError


class TransformerTokenizer(BaseTokenizer):
    """A tokenizer for Transformer.

    Args:
        tokenizer: A transformer tokenizer.
        tokenizer_args: Additional tokenizer arguments. Defaults to None.
    """

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
            "truncation": True,
        }

        if not self.__tokenizer_args.get("return_offsets_mapping", False):
            raise ValueError("Set return_offsets_mapping to True")

        if self.__tokenizer_args.get("return_tensors", "") != "pt":
            raise ValueError("Set return_tensors to pt")

    def __call__(self, texts: tuple[str, ...]) -> TextBatch:
        """Tokenize given texts, encode to tensors and return an instance of TextBatch.

        Args:
            texts: A tuple of strings, where each item represents a text.

        Returns:
            An instance of TextBatch.
        """
        batch_encoding = self.__tokenizer(texts, **self.__tokenizer_args)

        mappings = batch_encoding.pop("offset_mapping").tolist()

        pad_token_id = self.__tokenizer.pad_token_id

        mask = batch_encoding.input_ids != pad_token_id

        tokenized_text_lengths = mask.sum(dim=1)

        alignments = []
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

            alignments.append(
                Alignment(char_spans=char_spans, token_indices=tuple(token_indices))
            )

        return TextBatch(
            tagger_inputs=batch_encoding, mask=mask, alignments=tuple(alignments)
        )
