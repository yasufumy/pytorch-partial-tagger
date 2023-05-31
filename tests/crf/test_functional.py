from unittest.mock import patch

import pytest
import torch

from partial_tagger.crf import NINF
from partial_tagger.crf import functional as F

from .. import helpers


def test_log_likelihood_valid_as_probability(test_data_small: tuple) -> None:
    (batch_size, sequence_length, num_tags), log_potentials = test_data_small

    total_log_p = torch.tensor([NINF] * batch_size)
    for tag_indices in helpers.iterate_possible_tag_indices(sequence_length, num_tags):
        total_log_p = torch.logaddexp(
            total_log_p, F.log_likelihood(log_potentials, torch.tensor(tag_indices))
        )

    assert torch.allclose(total_log_p.exp(), torch.ones_like(total_log_p))


def test_marginal_log_likelihood_valid_as_probability(test_data_small: tuple) -> None:
    shape, log_potentials = test_data_small

    tag_bitmap = torch.ones(shape, dtype=torch.bool)
    log_p = F.marginal_log_likelihood(log_potentials, tag_bitmap)

    assert torch.allclose(log_p.exp(), torch.ones_like(log_p))


def test_marginal_log_likelihood_matches_log_likelihood_if_one_hot_tag_bitmap_is_given(
    test_data_small: tuple,
) -> None:
    shape, log_potentials = test_data_small

    for tag_bitmap in helpers.iterate_possible_one_hot_tag_bitmap(*shape):
        a = F.log_likelihood(log_potentials, tag_bitmap.long().argmax(dim=-1))
        b = F.marginal_log_likelihood(log_potentials, tag_bitmap)

        assert torch.allclose(a, b)


def test_forward_algorithm_returns_value_same_as_brute_force(
    test_data_small: tuple,
) -> None:
    _, log_potentials = test_data_small

    log_Z = F.forward_algorithm(log_potentials)
    expected_log_Z = helpers.compute_log_normalizer_by_brute_force(log_potentials)

    assert torch.allclose(log_Z, expected_log_Z)


def test_amax_returns_value_same_as_brute_force(test_data_small: tuple) -> None:
    _, log_potentials = test_data_small

    max_score = F.amax(log_potentials)
    expected_max_score, _ = helpers.compute_best_tag_indices_by_brute_force(
        log_potentials
    )

    assert torch.allclose(max_score, expected_max_score)


def test_decode_returns_value_same_as_brute_force(test_data_small: tuple) -> None:
    _, log_potentials = test_data_small

    max_score, tag_indices = F.decode(log_potentials)

    (
        expected_max_score,
        expected_tag_indices,
    ) = helpers.compute_best_tag_indices_by_brute_force(log_potentials)

    assert torch.allclose(max_score, expected_max_score)
    assert torch.allclose(tag_indices, expected_tag_indices)


def test_sequence_score_computes_mask_correctly(
    test_data_with_mask: tuple,
) -> None:
    _, log_potentials, tag_indices, mask = test_data_with_mask

    with patch("torch.Tensor.mul") as m:
        F.sequence_score(log_potentials, tag_indices, mask)
        used_mask = m.call_args[0][0]

        assert helpers.check_sequence_score_mask(used_mask, tag_indices, mask)


@pytest.mark.parametrize(
    "partial_index",
    list(range(5)),
)
def test_multitag_sequence_score_correctly_masks_log_potentials(
    test_data_with_mask: tuple, partial_index: int
) -> None:
    (_, _, num_tags), log_potentials, tag_indices, mask = test_data_with_mask
    tag_bitmap = F.to_tag_bitmap(tag_indices, num_tags, partial_index=partial_index)

    with patch("partial_tagger.crf.functional.forward_algorithm") as m:
        F.multitag_sequence_score(log_potentials, tag_bitmap, mask)
        constrained_log_potentials = m.call_args[0][0]

        assert helpers.check_constrained_log_potentials(
            log_potentials, constrained_log_potentials, tag_indices, mask, partial_index
        )


@pytest.mark.parametrize(
    "log_potentials, mask, start_constraints, end_constraints, transition_constraints",
    [
        (
            torch.randn(3, 20, 5, 5),
            torch.tensor(
                [
                    [True] * 20,
                    [True] * 15 + [False] * 5,
                    [True] * 8 + [False] * 12,
                ],
                dtype=torch.bool,
            ),
            torch.tensor([True, False, False, True, True]),  # 0, 3, 4 are allowed
            torch.tensor([False, True, True, False, False]),  # 2, 3 are allowed
            torch.tensor(
                [
                    [True, False, True, True, True],  # 0->1 is not allowed
                    [True, True, True, True, True],  # no constraints
                    [True, True, False, True, True],  # 2->2 is not allowed
                    [True, False, True, True, True],  # 3->1 is not allowed
                    [True, False, False, False, False],  # only 4->0 is allowed
                ]
            ),
        )
    ],
)
def test_constrains_log_potentials(
    log_potentials: torch.Tensor,
    mask: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
) -> None:
    constrained_log_potentials = F.constrain_log_potentials(
        log_potentials, mask, start_constraints, end_constraints, transition_constraints
    )
    lengths = mask.sum(dim=-1)

    for i, log_potential in enumerate(constrained_log_potentials):
        # Check start constraints
        for j, valid in enumerate(start_constraints):
            assert torch.equal(
                log_potential[0, j, j],
                log_potentials[i, 0, j, j] if valid else torch.tensor(NINF),
            )

        # Check end constraints
        for j, valid_e in enumerate(end_constraints):
            end_index = lengths[i] - 1
            if valid_e:
                for k, valid_t in enumerate(transition_constraints[:, j]):
                    assert torch.equal(
                        log_potential[end_index, k, j],
                        log_potentials[i, end_index, k, j]
                        if valid_t
                        else torch.tensor(NINF),
                    )
            else:
                assert torch.all(log_potential[end_index, :, j] == NINF)

        # Check transition constraints
        for pos in range(1, lengths[i] - 1):
            for j, row in enumerate(transition_constraints):
                for k, valid in enumerate(row):
                    assert torch.equal(
                        log_potential[pos, j, k],
                        log_potentials[i, pos, j, k] if valid else torch.tensor(NINF),
                    )


@pytest.mark.parametrize(
    "log_potentials, mask, start_constraints, end_constraints, transition_constraints",
    [
        (
            torch.randn(3, 20, 5, 5, requires_grad=True),
            torch.ones((3, 20), dtype=torch.bool),
            torch.tensor([True, False, False, True, True]),  # 0, 3, 4 are allowed
            torch.tensor([False, True, True, False, False]),  # 2, 3 are allowed
            torch.tensor(
                [
                    [True, False, True, True, True],  # 0->1 is not allowed
                    [True, True, True, True, True],  # no constraints
                    [True, True, False, True, True],  # 2->2 is not allowed
                    [True, False, True, True, True],  # 3->1 is not allowed
                    [True, False, False, False, False],  # only 4->0 is allowed
                ]
            ),
        ),
        (
            torch.zeros(3, 20, 5, 5, requires_grad=True),
            torch.ones((3, 20), dtype=torch.bool),
            torch.tensor([False, False, True, True, True]),
            torch.tensor([False, False, True, True, True]),
            torch.tensor([[True] * 5] * 5),
        ),
    ],
)
def test_constrained_decode_returns_expected_tag_indices_under_constraints(
    log_potentials: torch.Tensor,
    mask: torch.Tensor,
    start_constraints: torch.Tensor,
    end_constraints: torch.Tensor,
    transition_constraints: torch.Tensor,
) -> None:
    constrained_log_potentials = F.constrain_log_potentials(
        log_potentials, mask, start_constraints, end_constraints, transition_constraints
    )
    _, tag_indices = F.decode(constrained_log_potentials)

    assert helpers.check_tag_indices_satisfies_constraints(
        tag_indices, start_constraints, end_constraints, transition_constraints
    )


@pytest.mark.parametrize(
    "tag_indices, num_tags, expected, partial_index",
    [
        (
            torch.tensor([[0, 1, 2, 3, 4]]),
            5,
            torch.tensor(
                [
                    [
                        [True, False, False, False, False],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                    ]
                ]
            ),
            None,
        ),
        (torch.tensor([-100, -1, 5, 100]), 5, torch.tensor([[[False] * 5] * 4]), None),
        (
            torch.tensor([[0, 1, 2, 3, 4, -1, -1]]),
            5,
            torch.tensor(
                [
                    [
                        [True, False, False, False, False],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ]
            ),
            -1,
        ),
        (
            torch.tensor([[4, 1, 2, 3, 4, 0, 0]]),
            5,
            torch.tensor(
                [
                    [
                        [False, False, False, False, True],
                        [False, True, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, True, False],
                        [False, False, False, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ]
            ),
            0,
        ),
    ],
)
def test_tag_bitmap_returns_expected_value(
    tag_indices: torch.Tensor, num_tags: int, expected: torch.Tensor, partial_index: int
) -> None:
    assert torch.equal(F.to_tag_bitmap(tag_indices, num_tags, partial_index), expected)
