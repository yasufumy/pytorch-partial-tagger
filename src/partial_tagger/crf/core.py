from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from partial_tagger.crf.semiring import LogSemiring, MaxSemiring, Semiring, reduce


class BaseLogPartitions(metaclass=ABCMeta):
    @property
    @abstractmethod
    def value(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def marginals(self) -> torch.Tensor:
        raise NotImplementedError()


class UnitLogPartitions(BaseLogPartitions):
    def __init__(self, logits: torch.Tensor):
        self.__logits = logits

    @property
    def value(self) -> torch.Tensor:
        return LogSemiring.sum(self.__logits, dim=-1).squeeze(dim=-1)

    @property
    def marginals(self) -> torch.Tensor:
        return torch.log_softmax(self.__logits, dim=-1).exp()


class LogPartitions(BaseLogPartitions):
    def __init__(
        self,
        start_potentials: torch.Tensor,
        potentials: torch.Tensor,
        log_partitions: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.__start_potentials = start_potentials
        self.__potentials = potentials
        self.__log_partitions = log_partitions
        self.__mask = mask

    @property
    def value(self) -> torch.Tensor:
        return self.__log_partitions

    @property
    def marginals(self) -> torch.Tensor:
        (start, marginals) = torch.autograd.grad(
            self.__log_partitions.sum(),
            (self.__start_potentials, self.__potentials),
            create_graph=True,
        )
        return (
            torch.cat([start[:, None, :], marginals.sum(dim=-1)], dim=1)
            * self.__mask[..., None]
        )


class BaseCrfDistribution(metaclass=ABCMeta):
    def log_likelihood(self, tag_indices: torch.Tensor) -> torch.Tensor:
        return self.log_scores(tag_indices=tag_indices) - self.log_partitions.value

    def marginal_log_likelihood(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        return (
            self.log_multitag_scores(tag_bitmap=tag_bitmap) - self.log_partitions.value
        )

    @property
    def marginals(self) -> torch.Tensor:
        return self.log_partitions.marginals

    @abstractmethod
    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_partitions(self) -> BaseLogPartitions:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max(sefl) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def argmax(self, max_scores: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError()


class CrfUnitDistribution(BaseCrfDistribution):
    def __init__(self, logits: torch.Tensor):
        self.__logits = logits

    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        return (
            self.__logits.squeeze(dim=1)
            .gather(dim=-1, index=tag_indices)
            .squeeze(dim=-1)
        )

    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        logits = self.__logits.masked_fill(~tag_bitmap, Semiring.zero)
        return LogSemiring.sum(logits, dim=-1).squeeze(dim=-1)

    @property
    def log_partitions(self) -> UnitLogPartitions:
        return UnitLogPartitions(logits=self.__logits)

    @property
    def max(self) -> torch.Tensor:
        return MaxSemiring.sum(self.__logits, dim=-1).squeeze(dim=-1)

    @property
    def argmax(self) -> torch.Tensor:
        return torch.max(self.__logits, dim=-1).indices


class CrfDistribution(BaseCrfDistribution):
    def __init__(
        self,
        start_potentials: torch.Tensor,
        potentials: torch.Tensor,
        mask: torch.Tensor,
    ):
        self.__start_potentials = start_potentials
        self.__potentials = potentials
        self.__mask = mask

        self.__batch_size = potentials.size(0)
        self.__sequence_length = potentials.size(1)
        self.__num_tags = potentials.size(2)

    def log_scores(self, tag_indices: torch.Tensor) -> torch.Tensor:
        log_scores = self.__start_potentials.gather(
            index=tag_indices[:, [0]], dim=-1
        ).squeeze(dim=-1)
        log_scores += (
            self.__potentials.take_along_dim(
                indices=tag_indices[:, 1:, None, None], dim=-1
            )
            .take_along_dim(indices=tag_indices[:, :-1, None, None], dim=-2)
            .squeeze(dim=(-1, -2))
            * self.__mask[:, 1:]
        ).sum(dim=-1)
        return log_scores

    def log_multitag_scores(self, tag_bitmap: torch.Tensor) -> torch.Tensor:
        # Make sure to deactivate masked indices
        tag_bitmap = tag_bitmap & self.__mask[..., None]
        # Create transition mask
        mask = tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]
        # Flip masked indices. No need to touch them.
        mask |= (~self.__mask)[:, 1:, None, None]
        potentials = self.__potentials * mask + Semiring.zero * ~mask
        # Same as log_partitions
        transitions = reduce(semiring=LogSemiring, potentials=potentials)
        start_potentials = self.__start_potentials.masked_fill(
            ~tag_bitmap[:, 0], Semiring.zero
        )
        transitions = transitions + start_potentials[..., None]
        return LogSemiring.sum(LogSemiring.sum(transitions, dim=-1), dim=-1)

    @property
    def log_partitions(self) -> LogPartitions:
        transitions = reduce(semiring=LogSemiring, potentials=self.__potentials)
        transitions = transitions + self.__start_potentials[..., None]
        return LogPartitions(
            start_potentials=self.__start_potentials,
            potentials=self.__potentials,
            log_partitions=LogSemiring.sum(
                LogSemiring.sum(transitions, dim=-1), dim=-1
            ),
            mask=self.__mask,
        )

    @property
    def max(self) -> torch.Tensor:
        transitions = reduce(semiring=MaxSemiring, potentials=self.__potentials)
        transitions = transitions + self.__start_potentials[..., None]
        return MaxSemiring.sum(MaxSemiring.sum(transitions, dim=-1), dim=-1)

    @property
    def argmax(self) -> torch.Tensor:
        (transition_sequence,) = torch.autograd.grad(
            self.max.sum(),
            self.__potentials,
            create_graph=True,
        )
        transition_sequence = transition_sequence.long()

        tag_bitmap = transition_sequence.sum(dim=-2)
        tag_indices = tag_bitmap.argmax(dim=-1)

        start = transition_sequence[:, 0].sum(dim=-1).argmax(dim=-1, keepdim=True)

        return torch.cat([start, tag_indices], dim=-1)


class Crf(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_constraints: torch.Tensor | None = None,
        end_constraints: torch.Tensor | None = None,
        transition_constraints: torch.Tensor | None = None,
    ) -> BaseCrfDistribution:
        if mask is None:
            mask = logits.new_ones(logits.shape[:-1], dtype=torch.bool)

        batch_size, sequence_length, num_tags = logits.size()

        # Apply constrains
        if start_constraints is not None:
            logits[:, 0].masked_fill_(start_constraints, Semiring.zero)

        if end_constraints is not None:
            batch_indices = torch.arange(batch_size, device=logits.device)
            end_indices = mask.sum(dim=-1) - 1
            logits[batch_indices, end_indices] = logits[
                batch_indices, end_indices
            ].masked_fill(end_constraints, Semiring.zero)

        if sequence_length == 1:
            return CrfUnitDistribution(logits=logits)

        if transition_constraints is not None:
            transitions = self.transitions.masked_fill(
                transition_constraints, Semiring.zero
            )
        else:
            transitions = self.transitions

        logits = logits * mask[..., None]  # + Semiring.one * ~mask[..., None]
        mask_expanded = mask[:, 1:, None, None]
        transitions = transitions[None, None] * mask_expanded  # + Semiring.one * ~mask
        potentials = Semiring.mul(logits[:, 1:, None, :], transitions)

        # Masking
        mask_value = Semiring.eye(n=num_tags, dtype=logits.dtype, device=logits.device)
        potentials = potentials * mask_expanded + mask_value * (~mask_expanded)

        return CrfDistribution(
            start_potentials=logits[:, 0], potentials=potentials, mask=mask
        )
