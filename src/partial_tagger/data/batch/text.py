from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from ..core import Span, TokenizedText

Texts = Tuple[str, ...]
TaggerInputs = Dict[str, torch.Tensor]


class TokenizedTexts:
    def __init__(
        self,
        tokenized_texts: tuple[TokenizedText, ...],
        tagger_inputs: TaggerInputs,
        mask: torch.Tensor,
    ):
        self.tokenized_texts = tokenized_texts
        self.__tagger_inputs = tagger_inputs
        self.__mask = mask

    @property
    def size(self) -> int:
        return len(self.tokenized_texts)

    def get_tagger_inputs(self, device: torch.device) -> TaggerInputs:
        return {key: x.to(device) for key, x in self.__tagger_inputs.items()}

    def get_mask(self, device: torch.device) -> torch.Tensor:
        return self.__mask.to(device)


class BaseTokenizer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, texts: Texts) -> TokenizedTexts:
        raise NotImplementedError


class TransformerTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        tokenizer_args: dict | None = None,
    ):
        if not tokenizer.is_fast:
            raise ValueError("Only fast tokenizer is supported.")

        self.__tokenizer = tokenizer

        self.__tokenizer_args = tokenizer_args or {
            "padding": True,
            "return_tensors": "pt",
        }
        self.__tokenizer_args["return_offsets_mapping"] = True

    def __call__(self, texts: Texts) -> TokenizedTexts:
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
        return TokenizedTexts(tuple(tokenized_texts), batch_encoding, mask)
