from typing import cast

import torch
from torch.utils.data import DataLoader

from .data import LabelSet
from .data.batch.tag import CharBasedTagsBatch, TagFactory
from .data.batch.text import BaseTokenizer, TextBatch, Texts
from .tagger import SequenceTagger


class Recognizer:
    def __init__(
        self,
        tagger: SequenceTagger,
        tokenizer: BaseTokenizer,
        label_set: LabelSet,
        padding_index: int,
    ):
        self.__tagger = tagger
        self.__tokenizer = tokenizer
        self.__label_set = label_set
        self.__padding_index = padding_index

    def __call__(
        self, texts: Texts, batch_size: int, device: torch.device
    ) -> CharBasedTagsBatch:
        dataloader = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__tokenizer,
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval()

        predictions = []
        for text_batch in dataloader:
            text_batch = cast(TextBatch, text_batch)

            tag_indices = tagger.predict(
                text_batch.get_tagger_inputs(device),
                text_batch.get_mask(device),
            )

            tag_factory = TagFactory(text_batch.tokenized_texts, self.__label_set)
            predictions.extend(
                tag_factory.create_char_based_tags(tag_indices, self.__padding_index)
            )

        return tuple(predictions)
