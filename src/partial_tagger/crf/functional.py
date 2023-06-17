from typing import Callable, Optional, Tuple

import torch

from partial_tagger.crf import NINF

# collections.abc.Callable is preferred.
Matmul = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def log_likelihood(
    log_potentials: torch.Tensor,
    tag_indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes log likelihood.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        tag_indices: A [batch_size, sequence_length] integer tensor
            indicating an active index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing log likelihood.
    """

    score = sequence_score(log_potentials, tag_indices, mask)
    log_Z = forward_algorithm(log_potentials)

    return score - log_Z


def marginal_log_likelihood(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes marginal log likelihood.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
            indicating all active tags at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing marginal log likelihood.
    """

    score = multitag_sequence_score(log_potentials, tag_bitmap, mask)
    log_Z = forward_algorithm(log_potentials)

    return score - log_Z


def normalize(
    log_potentials: torch.Tensor, matmul: Matmul, normalizer: Callable
) -> torch.Tensor:
    """Normalizes log potentials based on normalizer.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        matmul: A general matrix multiplication.
        normalizer: A reduce operation.

    Returns:
        A [batch_size] float tensor representing the normalized value.
    """
    batch_size, sequence_length, num_tags, _ = log_potentials.size()

    n = sequence_length.bit_length()
    padding_length = (1 << n) - sequence_length
    padding_value = (
        1 - torch.eye(num_tags, num_tags, device=log_potentials.device)
    ).mul(NINF)[None, None]

    log_potentials = torch.cat(
        (
            log_potentials,
            padding_value.repeat(batch_size, padding_length, 1, 1),
        ),
        dim=1,
    )

    for _ in range(n):
        log_potentials = matmul(log_potentials[:, 0::2], log_potentials[:, 1::2])

    return normalizer(normalizer(log_potentials, dim=-2), dim=-1).squeeze(dim=-1)


def log_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes log-space matrix multiplication. This method computes logsumexp-sum
    operation instead of sum-prod operation (ordinary matmul). This computation is
    numerical stable.

    Args:
        a: a log-space tensor.
        b: a log-space tensor.

    Returns:
        a computed tensor.
    """
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def max_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes log-space max-sum operation instead of sum-prod operation
    (ordinary matmul). This computation is numerical stable.

    Args:
        a: a log-space tensor.
        b: a log-space tensor.

    Returns:
        a computed tensor.
    """
    return torch.max(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2).values


def forward_algorithm(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the normalizer for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.

    Returns:
        A [batch_size] float tensor representing the normalizer.
    """
    return normalize(log_potentials, log_matmul, torch.logsumexp)


def amax(log_potentials: torch.Tensor) -> torch.Tensor:
    """Computes the maximum score for a CRF.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.

    Returns:
        A [batch_size] float tensor representing the maximum score.
    """

    def _amax(inputs: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.max(inputs, dim=dim).values

    return normalize(log_potentials, max_matmul, _amax)


def decode(log_potentials: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the tag sequence gives the maximum probability for log potentials.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.

    Returns:
        A tuple of tensors. The first tensor is a [batch_size] float tensor
        representing the maximum score. The second tensor is
        a [batch_size, sequence_length] integer tensor representing the tag sequence.
    """
    max_score = amax(log_potentials)

    (tag_matrix,) = torch.autograd.grad(max_score.sum(), log_potentials)
    tag_matrix = tag_matrix.long()

    tag_bitmap = tag_matrix.sum(dim=-2)

    tag_indices = tag_bitmap.argmax(dim=-1)

    return max_score, tag_indices


def constrain_log_potentials(
    log_potentials: torch.Tensor,
    mask: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
) -> torch.Tensor:
    """Constrains start/end/transition to the given log potentials.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        mask: A [batch_size, sequence_length] boolean tensor.
        start_constraints: A [num_tags] boolean tensor.
        end_constraints: A [num_tags] boolean tensor.
        transition_constraints: A [num_tags, num_tags] boolean tensor.

    Returns:
        A [batch_size, sequence_length, num_tags, num_tags] float tensor.
    """
    # Apply transition constraints
    transition_constraints = transition_constraints | (~mask[..., None, None])
    constrained_log_potentials = log_potentials.masked_fill(
        ~transition_constraints, NINF
    )

    # Apply start constraints
    num_tags = log_potentials.size(-1)
    start_constraints = (
        start_constraints
        & torch.eye(num_tags, num_tags, device=log_potentials.device).bool()
    )
    constrained_log_potentials[:, 0] = log_potentials[:, 0].masked_fill(
        ~start_constraints, NINF
    )

    # Apply end constraints
    batch_indices = torch.arange(log_potentials.size(0), device=log_potentials.device)
    end_indices = mask.sum(dim=-1) - 1
    constrained_log_potentials[batch_indices, end_indices] = constrained_log_potentials[
        batch_indices, end_indices
    ].masked_fill_(~end_constraints, NINF)

    return constrained_log_potentials


def sequence_score(
    log_potentials: torch.Tensor,
    tag_indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the sequence score based on the given tag_bitmap.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        tag_indices: A [batch_size, sequence_length] integer tensor
            indicating an active index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing the sequence score.
    """
    if mask is None:
        mask = torch.ones_like(tag_indices, dtype=torch.bool)

    num_tags = log_potentials.size(-1)

    tag_bitmap = to_tag_bitmap(tag_indices, num_tags) & mask[..., None]

    initial_tag_matrix = (
        tag_bitmap[:, [0], :, None]
        & torch.eye(num_tags, num_tags, device=log_potentials.device).bool()
    )
    tag_matrix = torch.cat(
        (initial_tag_matrix, tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]),
        dim=1,
    )

    return log_potentials.mul(tag_matrix).sum(dim=(1, 2, 3))


def multitag_sequence_score(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the sequence score of all tag sequences matching.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tags, num_tags]
            float tensor.
        tag_bitmap: A [batch_size, sequence_length, num_tags] boolean tensor
            indicating all active tags at each index.
        mask: A [batch_size, sequence_length] boolean tensor.

    Returns:
        A [batch_size] float tensor representing the sequence score.
    """
    if mask is None:
        mask = tag_bitmap.new_ones(tag_bitmap.shape[:-1], dtype=torch.bool)

    num_tags = log_potentials.size(-1)

    tag_bitmap = tag_bitmap & mask[..., None]

    initial_tag_matrix = (
        tag_bitmap[:, [0], :, None]
        & torch.eye(num_tags, num_tags, device=log_potentials.device).bool()
    )
    tag_matrix = torch.cat(
        (initial_tag_matrix, tag_bitmap[:, :-1, :, None] & tag_bitmap[:, 1:, None, :]),
        dim=1,
    )
    tag_matrix |= (~mask)[..., None, None]

    constrained_log_potentials = log_potentials * tag_matrix + NINF * (~tag_matrix)
    return forward_algorithm(constrained_log_potentials)


def to_tag_bitmap(
    tag_indices: torch.Tensor, num_tags: int, partial_index: Optional[int] = None
) -> torch.Tensor:
    """Computes tag_bitmap from the given tag_indices.

    Args:
        tag_indices: A [batch_size, sequence_length] integer tensor.
        num_tags: An integer value representing the number of tags.
        partial_index: An integer value representing the index for partial label.

    Returns:
        A [batch_size, sequence_length, num_tags] boolean tensor.
        indicating an active tag at each index.
    """
    tag_bitmap = torch.arange(num_tags, device=tag_indices.device)[None, None].eq(
        tag_indices[..., None]
    )

    if partial_index is None:
        return tag_bitmap

    partial_mask = tag_indices.eq(partial_index)
    return tag_bitmap | partial_mask[..., None]
