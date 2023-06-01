from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch.nn import Module, Parameter

from ..crf import functional as F


class BaseConstrainer(Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, log_potentials: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Contrainer(BaseConstrainer):
    def __init__(
        self,
        start_states: list[bool],
        end_states: list[bool],
        transitions: list[list[bool]],
    ):
        super(Contrainer, self).__init__()

        self.start_states = Parameter(torch.tensor(start_states), requires_grad=False)
        self.end_states = Parameter(torch.tensor(end_states), requires_grad=False)
        self.transitions = Parameter(torch.tensor(transitions), requires_grad=False)

    def forward(self, log_potentials: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.constrain_log_potentials(
            log_potentials, mask, self.start_states, self.end_states, self.transitions
        )


class ViterbiDecoder(Module):
    """A Viterbi decoder for CRF.

    Args:
        padding_index: An integer for padded elements.
    """

    def __init__(
        self,
        padding_index: Optional[int] = -1,
        constrainer: Optional[BaseConstrainer] = None,
    ) -> None:
        super(ViterbiDecoder, self).__init__()

        self.padding_index = padding_index
        self.contrainer = constrainer

    def forward(self, log_potentials: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes the best tag sequence from the given log potentials.

        Args:
            log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length] integer tensor representing
            the best tag sequence.
        """

        if self.contrainer is not None:
            log_potentials = self.contrainer(log_potentials, mask)

        log_potentials.requires_grad_()

        with torch.enable_grad():
            _, tag_indices = F.decode(log_potentials)

        return tag_indices * mask + self.padding_index * (~mask)
