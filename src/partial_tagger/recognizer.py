from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from partial_tagger.data import Alignments, LabelSet, Tag
from partial_tagger.data.collators import BaseCollator, Batch
from partial_tagger.tagger import SequenceTagger


class Recognizer:
    """A recognizer which predicts character-based tags from a given text with
    a trained sequence tagger.

    Args:
        tagger: An instance of SequenceTagger representing the trained tagger.
        collator: Any instance of the classes that inherit BaseCollator for
            encoding given texts into tensors.
        label_set: An instance of LabelSet.
    """

    def __init__(
        self,
        tagger: SequenceTagger,
        collator: BaseCollator,
        label_set: LabelSet,
    ):
        self.__tagger = tagger
        self.__collator = collator
        self.__label_set = label_set

    def __call__(
        self, texts: tuple[str, ...], batch_size: int, device: torch.device
    ) -> tuple[set[Tag], ...]:
        """Predicts character-based tags from given texts using a trained tagger.

        Args:
            texts: A tuple of input texts.
            batch_size: An integer representing a batch size.
            device: The device to use for prediction.

        Returns:
            A tuple where each item is a set of predicted character-based tags
            for each input text.

        """
        dataloader: Sequence[tuple[Batch, Alignments]] = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__collator,
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval().to(device)

        predictions = []
        for batch, alignments in dataloader:
            batch = batch.to(device)

            tag_indices = tagger.predict(batch.tagger_inputs, batch.mask)

            predictions.extend(
                alignments.create_char_based_tags(
                    tag_indices=tag_indices.tolist(),
                    label_set=self.__label_set,
                    padding_index=tagger.padding_index,
                )
            )

        return tuple(predictions)
