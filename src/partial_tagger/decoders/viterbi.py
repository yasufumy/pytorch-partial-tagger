from __future__ import annotations

import torch
from torch.nn import Module, Parameter

from partial_tagger.crf import functional as F


class Constrainer(Module):
    """A constrainer to constrain the given log potentials of a CRF.

    Args:
        start_states: A list of boolean values representing the start states.
            True indicates an allowed state, while False indicates an otherwise state.
        end_states: A list of boolean values representing the end states.
            True indicates an allowed state, while False indicates an otherwise state.
        transitions: A list of lists of boolean values representing the transitions.
            True indicates an allowed transition,
            while False indicates an otherwise transition.

    Attributes:
        start_states: Start states parameters.
        end_states: End states parameters.
        transitions: transitions parameters.

    """

    def __init__(
        self,
        start_states: list[bool],
        end_states: list[bool],
        transitions: list[list[bool]],
    ):
        super(Constrainer, self).__init__()

        self.start_states = Parameter(torch.tensor(start_states), requires_grad=False)
        self.end_states = Parameter(torch.tensor(end_states), requires_grad=False)
        self.transitions = Parameter(torch.tensor(transitions), requires_grad=False)

    def forward(self, log_potentials: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies constraints to a given log potentials.

        Args:
            log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
                float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, num_tags, num_tags] float tensor
            representing constrained log potentials.
        """
        return F.constrain_log_potentials(
            log_potentials, mask, self.start_states, self.end_states, self.transitions
        )


class ViterbiDecoder(Module):
    """A Viterbi decoder for a CRF.

    Args:
        padding_index: An integer for padded elements.
        constrainer: A instance of Constrainer to constrain a given log potentials.

    Attributes:
        padding_index: An integer for padded elements.
        constrainer: A constrainer.
    """

    def __init__(
        self,
        padding_index: int = -1,
        constrainer: Constrainer | None = None,
    ) -> None:
        super(ViterbiDecoder, self).__init__()

        self.padding_index = padding_index
        self.constrainer = constrainer

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

        if self.constrainer is not None:
            log_potentials = self.constrainer(log_potentials, mask)

        log_potentials.requires_grad_()

        with torch.enable_grad():
            _, tag_indices = F.decode(log_potentials)

        return tag_indices * mask + self.padding_index * (~mask)
