from itertools import product
from typing import Generator, Tuple

import torch

from partial_tagger.crf import NINF
from partial_tagger.crf import functional as F


def iterate_possible_tag_indices(
    sequence_length: int, num_tags: int
) -> Generator[tuple, None, None]:
    yield from product(range(num_tags), repeat=sequence_length)


def iterate_possible_one_hot_tag_bitmap(
    batch_size: int, sequence_length: int, num_tags: int
) -> Generator[torch.Tensor, None, None]:
    for tag_indices in iterate_possible_tag_indices(sequence_length, num_tags):
        tag_bitmap = []
        for active in tag_indices:
            bitmap = [False] * num_tags
            bitmap[active] = True
            tag_bitmap.append(bitmap)
        yield torch.tensor([tag_bitmap] * batch_size)


def compute_log_normalizer_by_brute_force(log_potentials: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, num_tags, _ = log_potentials.size()
    log_Z = torch.tensor([NINF] * batch_size)
    for b in range(batch_size):
        for tag_indices in iterate_possible_tag_indices(sequence_length + 1, num_tags):
            tag_indices_score = torch.tensor(0.0)
            for i, (j, k) in enumerate(zip(tag_indices[:-1], tag_indices[1:])):
                tag_indices_score += log_potentials[b, i, j, k]
            log_Z[b] = torch.logaddexp(log_Z[b], tag_indices_score)
    return log_Z


def compute_best_tag_indices_by_brute_force(
    log_potentials: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, num_tags, _ = log_potentials.size()
    best_tag_indices = torch.tensor([[-1] * sequence_length for _ in range(batch_size)])
    max_scores = torch.tensor([NINF] * batch_size)
    for b in range(batch_size):
        max_score = torch.tensor(NINF)
        for tag_indices in iterate_possible_tag_indices(sequence_length, num_tags):
            tag_indices_score = (
                log_potentials[b, 0, tag_indices[0], tag_indices[0]].detach().clone()
            )
            for i, (j, k) in enumerate(zip(tag_indices[:-1], tag_indices[1:]), 1):
                tag_indices_score += log_potentials[b, i, j, k]
            if tag_indices_score.gt(max_score):
                # Ignore the dummy initial state
                best_tag_indices[b] = torch.tensor(tag_indices)
                max_score = tag_indices_score
        max_scores[b] = max_score
    return max_scores, best_tag_indices


def check_tag_indices_satisfies_constraints(
    tag_indices: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
) -> bool:
    sequence_length = tag_indices.size(-1)
    for tags in tag_indices:
        if not start_constraints[tags[0]]:
            return False
        if not end_constraints[tags[-1]]:
            return False
        for i in range(sequence_length - 1):
            if not transition_constraints[tags[i], tags[i + 1]]:
                return False
    return True


def check_log_potentials_masked_correctly(
    log_potentials: torch.Tensor, mask: torch.Tensor
) -> bool:
    sequence_length = log_potentials.size(1)
    num_tags = log_potentials.size(-1)
    mask_value = (~torch.eye(num_tags, num_tags).bool()).mul(NINF)
    lengths = mask.sum(dim=-1)
    for b, real_sequence_length in enumerate(lengths):
        for L in range(real_sequence_length, sequence_length):
            if not torch.allclose(log_potentials[b, L], mask_value):
                return False
    return True


def check_log_potentials_behaves_as_probability(log_potentials: torch.Tensor) -> bool:
    batch_size = log_potentials.size(0)
    sequence_length = log_potentials.size(1)
    num_tags = log_potentials.size(2)
    log_probability = torch.tensor([NINF] * batch_size)
    for tag_indices in iterate_possible_tag_indices(sequence_length, num_tags):
        log_probability = torch.logaddexp(
            log_probability, F.log_likelihood(log_potentials, torch.tensor(tag_indices))
        )
    return torch.allclose(log_probability.exp(), torch.ones_like(log_probability))


def check_sequence_score_mask(
    used_mask: torch.Tensor, tag_indices: torch.Tensor, mask: torch.Tensor
) -> bool:
    num_tags = used_mask.size(-1)
    lengths = mask.sum(dim=-1)
    for b, real_sequence_length in enumerate(lengths):
        # only (i, j) is True, otherwise False
        tags = tag_indices[b, :real_sequence_length].tolist()
        tags = [tags[0]] + tags
        for pos, (i, j) in enumerate(zip(tags[:-1], tags[1:])):
            if not used_mask[b, pos, i, j]:
                return False
            for x in range(num_tags):
                for y in range(num_tags):
                    if x == i and y == j:
                        continue
                    if used_mask[b, pos, x, y]:
                        return False

        # all values should be False.
        ignores = used_mask[b, real_sequence_length:]
        if not torch.equal(ignores, torch.zeros_like(ignores, dtype=torch.bool)):
            return False
    return True


def check_constrained_log_potentials(
    log_potentials: torch.Tensor,
    constrained_log_potentials: torch.Tensor,
    tag_indices: torch.Tensor,
    mask: torch.Tensor,
    partial_index: int,
) -> bool:
    num_tags = log_potentials.size(-1)
    lengths = mask.sum(dim=-1)
    for b, real_sequence_length in enumerate(lengths):
        tags = tag_indices[b, :real_sequence_length].tolist()
        tags = [tags[0]] + tags
        for pos, (i, j) in enumerate(zip(tags[:-1], tags[1:])):
            if i == partial_index and j == partial_index:
                x = constrained_log_potentials[b, pos]
                y = log_potentials[b, pos]
            elif i == partial_index:
                for n in range(num_tags):
                    if n == j:
                        continue
                    temp = constrained_log_potentials[b, pos, :, n]
                    if not torch.equal(temp, torch.full_like(temp, NINF)):
                        return False
                x = constrained_log_potentials[b, pos, :, j]
                y = log_potentials[b, pos, :, j]
            elif j == partial_index:
                for m in range(num_tags):
                    if m == i:
                        continue
                    temp = constrained_log_potentials[b, pos, m]
                    if not torch.equal(temp, torch.full_like(temp, NINF)):
                        return False
                x = constrained_log_potentials[b, pos, i]
                y = log_potentials[b, pos, i]
            else:
                for m in range(num_tags):
                    for n in range(num_tags):
                        if m == i and n == j:
                            continue
                        if constrained_log_potentials[b, pos, m, n] != NINF:
                            return False
                x = constrained_log_potentials[b, pos, i, j]
                y = log_potentials[b, pos, i, j]

            if not torch.equal(x, y):
                return False

        x = constrained_log_potentials[b, real_sequence_length:]
        y = log_potentials[b, real_sequence_length:]
        if not torch.equal(x, y):
            return False
    return True
