from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from partial_tagger.data import Alignments, Tag


@dataclass
class Batch:
    tagger_inputs: dict[str, torch.Tensor]
    mask: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            tagger_inputs={
                key: tensor.to(device) for key, tensor in self.tagger_inputs.items()
            },
            mask=self.mask.to(device),
        )


class BaseCollator(metaclass=ABCMeta):
    """Base class for all collators."""

    @abstractmethod
    def __call__(self, texts: tuple[str, ...]) -> tuple[Batch, Alignments]:
        """Tokenizes given texts and encodes them into tensors. Also, provides an
        instance of Alignments based on the tokenization results.

        Args:
            texts: A tuple of strings where each item represents a text.

        Returns:
            A pair of instances of Batch and Alignments.
        """
        raise NotImplementedError


class TransformerCollator(BaseCollator):
    """A collator class for transformers.

    Args:
        tokenizer: A transformer tokenizer.
        tokenizer_args: Additional tokenizer arguments. Defaults to None.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_args: dict[str, Any] | None = None,
    ):
        if not tokenizer.is_fast:
            raise ValueError("Only transformers.PreTrainedTokenizerFast is supported.")

        self.__tokenizer = tokenizer
        self.__tokenizer_args = tokenizer_args or {
            "return_offsets_mapping": True,
            "truncation": True,
            "return_tensors": "pt",
        }

        if not self.__tokenizer_args.get("return_offsets_mapping", False):
            raise ValueError("Set return_offsets_mapping to True")

        if self.__tokenizer_args.get("return_tensors", "") != "pt":
            raise ValueError("Set return_tensors to pt")

        self.__tokenizer_args.pop("return_tensors")

    def __call__(self, texts: tuple[str, ...]) -> tuple[Batch, Alignments]:
        """Tokenizes given texts and encodes them into tensors. Also, provides an
        instance of Alignments based on the tokenization results.

        Args:
            texts: A tuple of strings where each item represents a text.

        Returns:
            A pair of instances of Batch and Alignments.
        """
        temp = self.__tokenizer(texts, **self.__tokenizer_args)

        alignments = Alignments.from_offset_mapping(
            offset_mapping=temp.pop("offset_mapping"),
            char_lengths=tuple(map(len, texts)),
        )

        batch_encoding = self.__tokenizer.pad(temp, return_tensors="pt")
        mask = batch_encoding.input_ids != self.__tokenizer.pad_token_id

        return Batch(tagger_inputs=batch_encoding, mask=mask), alignments


class TrainingCollator:
    """A thin wrapper class for any instance of the classes that inherit BaseCollator.

    Args:
        collator: Any instance the classes that inherit BaseCollator.
    """

    def __init__(self, collator: BaseCollator):
        self.__collator = collator

    def __call__(
        self, examples: list[tuple[str, set[Tag]]]
    ) -> tuple[Batch, Alignments, tuple[set[Tag], ...]]:
        texts, ground_truths = zip(*examples)

        batch, alignments = self.__collator(texts)

        return batch, alignments, ground_truths
