from __future__ import annotations

import torch

from ..core import CharBasedTags, LabelSet, TokenBasedTags


class TagsBatch:
    """A batch of token-based tags.

    Args:
        tags_batch: A tuple of instances of TokenBasedTags.
        label_set: An instance of LabelSet to use for tag conversion.
        device: A device on which to place tensors. Defaults to None.
    """

    def __init__(
        self,
        tags_batch: tuple[TokenBasedTags, ...],
        label_set: LabelSet,
        device: torch.device | None = None,
    ):
        self.__tags_batch = tags_batch
        self.__label_set = label_set
        self.__device = device

    def to(self, device: torch.device) -> None:
        self.__device = device

    @property
    def size(self) -> int:
        return len(self.__tags_batch)

    @property
    def char_based(self) -> tuple[CharBasedTags, ...]:
        return tuple(tags.convert_to_char_based() for tags in self.__tags_batch)

    @property
    def token_based(self) -> tuple[TokenBasedTags, ...]:
        return self.__tags_batch

    def get_tag_indices(
        self, padding_index: int = -1, unknown_index: int = -100
    ) -> torch.Tensor:
        """Returns a tensor of tag indices for a batch.

        Args:
            padding_index: An integer representing an index to pad a tensor.
            Defaults to -1.
            unknown_index: An integer representing an index for an unknown tag.
            Defaults to -100.

        Returns:
            A [batch_size, sequence_length] integer tensor representing tag indices.
        """
        label_set = self.__label_set

        max_length = max(tags.num_tokens for tags in self.__tags_batch)

        tag_indices = []

        for tags in self.__tags_batch:
            indices = tags.get_tag_indices(label_set, unknown_index)
            tag_indices.append(indices + [padding_index] * (max_length - len(indices)))

        tensor = torch.tensor(tag_indices)

        if self.__device is not None:
            tensor = tensor.to(self.__device)

        return tensor

    def get_tag_bitmap(
        self,
    ) -> torch.Tensor:
        """Returns a tensor of tag bitmap for a batch.

        Returns:
            A [batch_size, sequence_length, num_tags] boolean tensor representing
            tag bitmap.
        """
        label_set = self.__label_set

        max_length = max(tags.num_tokens for tags in self.__tags_batch)

        tag_bitmap = []

        for tags in self.__tags_batch:
            bitmap = tags.get_tag_bitmap(label_set)
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
