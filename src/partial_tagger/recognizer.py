from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from sequence_label import LabelAlignment, LabelSet, SequenceLabel

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
    ) -> tuple[SequenceLabel, ...]:
        """Predicts character-based tags from given texts using a trained tagger.

        Args:
            texts: A tuple of input texts.
            batch_size: An integer representing a batch size.
            device: The device to use for prediction.

        Returns:
            A tuple where each item is a set of predicted character-based tags
            for each input text.

        """
        dataloader: Sequence[tuple[Batch, tuple[LabelAlignment, ...]]] = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__collator,  # type:ignore
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval().to(device)

        predictions: list[SequenceLabel] = []
        for batch, alignments in dataloader:
            batch = batch.to(device)

            tag_indices = tagger.predict(batch.tagger_inputs, batch.mask)

            predictions.extend(
                self.__label_set.decode(
                    tag_indices=tag_indices.tolist(), alignments=alignments
                )
            )

        return tuple(predictions)
