from typing import cast

import torch
from torch.utils.data import DataLoader

from .data.batch import BaseBatchFactory, Batch, CharBasedTagsCollection, Texts
from .tagger import SequenceTagger


class Recognizer:
    def __init__(
        self,
        tagger: SequenceTagger,
        batch_factory: BaseBatchFactory,
        padding_index: int,
    ):
        self.__tagger = tagger
        self.__batch_factory = batch_factory
        self.__padding_index = padding_index

    def __call__(
        self, texts: Texts, batch_size: int, device: torch.device
    ) -> CharBasedTagsCollection:
        dataloader = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__batch_factory.create,  # type:ignore
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval()

        predictions = []
        for batch in dataloader:
            batch = cast(Batch, batch)

            tag_indices = tagger.predict(
                batch.get_tagger_inputs(device), batch.get_mask(device)
            )

            predictions.extend(
                batch.create_char_based_tags(tag_indices, self.__padding_index)
            )

        return tuple(predictions)
