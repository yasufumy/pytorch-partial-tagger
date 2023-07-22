from __future__ import annotations

import torch

from partial_tagger.data.core import Alignment, LabelSet, Tag


class TagsBatch:
    """A batch of character-based tags

    Args:
        tags_batch: A tuple of sets of character-based tags.
        alignments: A tuple of instances of Alignment.
        device: A device on which to place tensors. Defaults to None.
    """

    def __init__(
        self,
        tags_batch: tuple[set[Tag], ...],
        alignments: tuple[Alignment, ...],
        device: torch.device | None = None,
    ):
        if len(tags_batch) != len(alignments):
            raise ValueError(
                "The length of tags_batch and alignments must be"
                f" the same: {len(tags_batch)} != {len(alignments)}"
            )

        if len(tags_batch) <= 0:
            raise ValueError(
                "The length of tags_batch must be greater than 1:" f" {len(tags_batch)}"
            )

        if len(alignments) <= 0:
            raise ValueError(
                "The length of alignments must be greater than 1:" f" {len(tags_batch)}"
            )

        self.__tags_batch = tags_batch
        self.__alignments = alignments
        self.__device = device

    def to(self, device: torch.device) -> None:
        self.__device = device

    @property
    def size(self) -> int:
        return len(self.__tags_batch)

    @property
    def char_based(self) -> tuple[set[Tag], ...]:
        return self.__tags_batch

    @property
    def token_based(self) -> tuple[set[Tag], ...]:
        return tuple(
            alignment.align_token_based(tags=tags)
            for tags, alignment in zip(self.__tags_batch, self.__alignments)
        )

    def get_tag_indices(
        self, label_set: LabelSet, padding_index: int = -1, unknown_index: int = -100
    ) -> torch.Tensor:
        """Returns a tensor of tag indices for a batch.

        Args:
            label_set: An instance of LabelSet.
            padding_index: An integer representing an index to pad a tensor.
                Defaults to -1.
            unknown_index: An integer representing an index for an unknown tag.
                Defaults to -100.

        Returns:
            A [batch_size, sequence_length] integer tensor representing tag indices.
        """
        max_length = max(alignment.num_tokens for alignment in self.__alignments)

        tag_indices = []

        for tags, alignment in zip(self.__tags_batch, self.__alignments):
            indices = alignment.create_tag_indices(
                tags=tags, label_set=label_set, unknown_index=unknown_index
            )
            tag_indices.append(indices + [padding_index] * (max_length - len(indices)))

        tensor = torch.tensor(tag_indices)

        if self.__device is not None:
            tensor = tensor.to(self.__device)

        return tensor

    def get_tag_bitmap(self, label_set: LabelSet) -> torch.Tensor:
        """Returns a tensor of tag bitmap for a batch.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            A [batch_size, sequence_length, num_tags] boolean tensor representing
            tag bitmap.
        """
        max_length = max(alignment.num_tokens for alignment in self.__alignments)

        tag_bitmap = []

        for tags, alignment in zip(self.__tags_batch, self.__alignments):
            bitmap = alignment.create_tag_bitmap(tags=tags, label_set=label_set)
            tag_bitmap.append(
                bitmap
                + [
                    [False] * label_set.get_tag_size()
                    for _ in range(max_length - len(bitmap))
                ]
            )

        tensor = torch.tensor(tag_bitmap)

        if self.__device:
            tensor = tensor.to(self.__device)

        return tensor
