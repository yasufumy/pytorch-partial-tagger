from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Final

import torch


class Semiring(metaclass=ABCMeta):
    zero: Final[float] = -5e3
    one: Final[float] = 0.0

    @classmethod
    def eye(cls, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        eye = torch.full(size=(n, n), fill_value=cls.zero, dtype=dtype, device=device)
        return torch.diagonal_scatter(
            eye, torch.full(size=(n,), fill_value=cls.one, dtype=dtype, device=device)
        )

    @staticmethod
    @abstractmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def prod(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.sum(tensor, dim=dim)

    @staticmethod
    @abstractmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    @classmethod
    def bmm(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cls.sum(cls.mul(x.unsqueeze(-1), y.unsqueeze(-3)), dim=-2)


class LogSemiring(Semiring):
    @staticmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(tensor, dim=dim)

    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(x, y)


class MaxSemiring(Semiring):
    @staticmethod
    def sum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.max(tensor, dim=dim).values

    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, y)


def reduce(semiring: type[Semiring], potentials: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_tags, _ = potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    padding_value = semiring.eye(
        n=num_tags, dtype=potentials.dtype, device=potentials.device
    )[None, None]

    potentials = torch.cat(
        (
            potentials,
            padding_value.repeat(batch_size, padding_length, 1, 1),
        ),
        dim=1,
    )

    for _ in range(n):
        potentials = semiring.bmm(potentials[:, 0::2], potentials[:, 1::2])

    return potentials.squeeze(dim=1)
