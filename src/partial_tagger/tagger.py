from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from sequence_classifier.crf import BaseCrfDistribution, Crf
from torch.nn import Module, Parameter

if TYPE_CHECKING:
    from partial_tagger.encoders.base import BaseEncoder


class SequenceTagger(Module):
    """A sequence tagging model with a CRF layer.

    Args:
        encoder: An encoder module.
        decoder: A decoder module.

    Attributes:
        encoder: An encoder module.
        crf: A CRF layer.
        decoder: A decoder module.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        padding_index: int,
        start_states: tuple[bool, ...] | None = None,
        end_states: tuple[bool, ...] | None = None,
        transitions: tuple[tuple[bool, ...], ...] | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.crf = Crf(encoder.get_hidden_size())
        self.__padding_index = padding_index
        self.start_constraints = (
            Parameter(~torch.tensor(start_states), requires_grad=False)
            if start_states is not None
            else None
        )
        self.end_constraints = (
            Parameter(~torch.tensor(end_states), requires_grad=False)
            if end_states is not None
            else None
        )
        self.transition_constraints = (
            Parameter(~torch.tensor(transitions), requires_grad=False)
            if transitions is not None
            else None
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        mask: torch.Tensor,
        constrain: bool = False,
    ) -> BaseCrfDistribution:
        """Computes log potentials and tag sequence.

        Args:
            inputs: An inputs representing input data feeding into the encoder module.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A pair of a [batch_size, sequence_length, num_tags, num_tags] float tensor
            and a [batch_size, sequence_length] integer tensor.
            The float tensor representing log potentials and
            the integer tensor representing tag sequence.
        """
        if constrain:
            dist = self.crf(
                logits=self.encoder(inputs),
                mask=mask,
                start_constraints=self.start_constraints,
                end_constraints=self.end_constraints,
                transition_constraints=self.transition_constraints,
            )
        else:
            dist = self.crf(logits=self.encoder(inputs), mask=mask)

        return cast(BaseCrfDistribution, dist)

    def predict(
        self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        dist = self(inputs=inputs, mask=mask, constrain=True)
        tag_indices = cast(BaseCrfDistribution, dist).argmax

        return cast(torch.Tensor, tag_indices * mask + self.__padding_index * (~mask))
