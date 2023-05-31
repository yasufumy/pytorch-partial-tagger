import pytest
import torch

from partial_tagger.crf.nn import CRF

from .. import helpers


@pytest.fixture
def crf(num_tags: int) -> CRF:
    return CRF(num_tags).eval()


def test_log_potentials_behaves_as_probability(crf: CRF, num_tags: int) -> None:
    batch_size = 12
    sequence_length = 3
    logits = torch.randn(batch_size, sequence_length, num_tags)

    with torch.no_grad():
        log_potentials = crf(logits)

    assert helpers.check_log_potentials_behaves_as_probability(log_potentials)


def test_crf_masks_log_potentials(crf: CRF, num_tags: int) -> None:
    batch_size = 3
    sequence_length = 20
    logits = torch.randn(batch_size, sequence_length, num_tags)
    mask = torch.tensor([[True] * 20, [True] * 10 + [False] * 10, [False] * 20])

    with torch.no_grad():
        log_potentials = crf(logits, mask)

    assert helpers.check_log_potentials_masked_correctly(log_potentials, mask)
