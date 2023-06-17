from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from partial_tagger.data import CharBasedTags, LabelSet
from partial_tagger.data.batch.text import BaseTokenizer, TextBatch
from partial_tagger.tagger import SequenceTagger


class Recognizer:
    """A recognizer which predicts character-based tags from a given text with
    a trained sequence tagger.

    Args:
        tagger: An instance of SequenceTagger representing the trained tagger.
        tokenizer: An instance of BaseTokenizer for tokenizing the input texts.
        label_set: An instance of LabelSet.
    """

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
        """Predicts character-based tags from given texts using a trained tagger.

        Args:
            texts: A tuple of input texts.
            batch_size: An integer representing a batch size.
            device: The device to use for prediction.

        Returns:
            A tuple of an instance of CharBasedTags representing predicted tags
            for each input text.

        """
        dataloader: Sequence[TextBatch] = DataLoader(
            texts,  # type: ignore
            collate_fn=self.__tokenizer,
            batch_size=batch_size,
            shuffle=False,
        )

        tagger = self.__tagger.eval().to(device)

        predictions = []
        for text_batch in dataloader:
            text_batch.to(device)

            tag_indices = tagger.predict(text_batch.tagger_inputs, text_batch.mask)

            predictions.extend(
                text_batch.create_char_based_tags(
                    tag_indices, self.__label_set, tagger.padding_index
                )
            )

        return tuple(predictions)
