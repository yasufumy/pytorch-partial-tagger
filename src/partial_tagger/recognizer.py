from __future__ import annotations

from typing import cast

import torch
from torch.utils.data import DataLoader

from .data import CharBasedTags, LabelSet
from .data.batch.text import BaseTokenizer, TextBatch
from .tagger import SequenceTagger


class Recognizer:
    def __init__(
        self,
        tagger: SequenceTagger,
        tokenizer: BaseTokenizer,
        label_set: LabelSet,
    ):
        self.__tagger = tagger
        self.__tokenizer = tokenizer
        self.__label_set = label_set

    def __call__(
        self, texts: tuple[str, ...], batch_size: int, device: torch.device
    ) -> tuple[CharBasedTags, ...]:
        dataloader = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__tokenizer,
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval().to(device)

        predictions = []
        for text_batch in dataloader:
            text_batch = cast(TextBatch, text_batch)

            text_batch.to(device)

            tag_indices = tagger.predict(text_batch.tagger_inputs, text_batch.mask)

            predictions.extend(
                text_batch.create_char_based_tags(
                    tag_indices, self.__label_set, tagger.padding_index
                )
            )

        return tuple(predictions)
