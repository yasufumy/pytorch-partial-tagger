import pytest
import torch

from partial_tagger.crf import NINF


@pytest.fixture
def num_tags() -> int:
    return 5


@pytest.fixture
def test_data_for_shape_check(num_tags: int) -> tuple:
    batch_size = 3
    sequence_length = 20
    embedding_size = 128
    embeddings = torch.randn(batch_size, sequence_length, embedding_size)
    logits = torch.randn(batch_size, sequence_length, num_tags)
    log_potentials = torch.randn(batch_size, sequence_length, num_tags, num_tags)
    tag_indices = torch.randint(0, num_tags, (batch_size, sequence_length))
    tag_bitmap = torch.nn.functional.one_hot(tag_indices, num_tags).bool()

    return (
        (batch_size, sequence_length, embedding_size, num_tags),
        embeddings,
        logits,
        log_potentials,
        tag_indices,
        tag_bitmap,
    )


@pytest.fixture
def test_data_small(num_tags: int) -> tuple:
    batch_size = 2
    sequence_length = 3
    log_potentials = torch.randn(batch_size, sequence_length, num_tags, num_tags)
    initial_mask = torch.eye(num_tags, num_tags).bool()
    log_potentials[:, 0] = log_potentials[:, 0] * initial_mask + NINF * (~initial_mask)
    log_potentials.requires_grad_()
    return (batch_size, sequence_length, num_tags), log_potentials


@pytest.fixture
def transitions() -> torch.Tensor:
    return torch.tensor(
        [
            # O   B-X  I-X  B-Y  I-Y
            [0.0, 1.0, 0.0, 0.0, 0.0],  # O
            [0.0, 0.0, 1.0, 0.0, 0.0],  # B-X
            [0.0, 0.0, 0.0, 1.0, 0.0],  # I-X
            [0.0, 0.0, 0.0, 0.0, 1.0],  # B-Y
            [1.0, 0.0, 0.0, 0.0, 0.0],  # I-Y
        ]
    )


@pytest.fixture
def test_data_by_hand(num_tags: int) -> tuple:
    batch_size = 2
    sequence_length = 3
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 100.0],
            ],
        ]
    )
    tag_indices = torch.tensor([[1, 2, 3], [3, 4, 4]])
    return (batch_size, sequence_length, num_tags), logits, tag_indices


@pytest.fixture
def test_data_with_mask(num_tags: int) -> tuple:
    batch_size = 3
    sequence_length = 20
    log_potentials = torch.randn(batch_size, sequence_length, num_tags, num_tags)
    # a dummy initial token
    initial_mask = torch.eye(num_tags, num_tags).bool()
    log_potentials[:, 0] = log_potentials[:, 0] * initial_mask + NINF * (~initial_mask)
    mask = torch.tensor(
        [
            [True] * (sequence_length - 2 * i) + [False] * 2 * i
            for i in range(batch_size)
        ]
    )
    tag_indices = torch.randint(0, num_tags, (batch_size, sequence_length))
    return (batch_size, sequence_length, num_tags), log_potentials, tag_indices, mask
