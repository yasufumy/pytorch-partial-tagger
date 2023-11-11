from __future__ import annotations

import torch
from hypothesis import given
from hypothesis import strategies as st
from sequence_classifier.crf import Crf
from sequence_label import LabelSet

from partial_tagger.crf import functional as F
from partial_tagger.crf.nn import CRF


@st.composite
def inputs1(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = draw(st.integers(min_value=1, max_value=5))
    sequence_length = 30

    lengths = []
    for _ in range(batch_size):
        lengths.append(draw(st.integers(min_value=2, max_value=sequence_length)))

    max_length = max(lengths)

    num_tags = 21
    logits = torch.randn(batch_size, max_length, num_tags, requires_grad=True)
    mask = torch.arange(max_length) < torch.tensor(lengths)[..., None]
    tag_indices = torch.randint(low=0, high=num_tags, size=(batch_size, max_length))
    return logits, mask, tag_indices


@st.composite
def inputs2(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = draw(st.integers(min_value=1, max_value=5))
    sequence_length = 30

    lengths = []
    for _ in range(batch_size):
        lengths.append(draw(st.integers(min_value=2, max_value=sequence_length)))

    max_length = max(lengths)

    num_tags = 21
    logits = torch.randn(batch_size, max_length, num_tags, requires_grad=True)
    mask = torch.arange(max_length) < torch.tensor(lengths)[..., None]
    tag_indices = torch.randint(low=0, high=num_tags, size=(batch_size, max_length))
    tag_bitmap = torch.randint(
        low=0, high=2, size=(batch_size, max_length, num_tags)
    ).bool()
    return logits, mask, F.to_tag_bitmap(tag_indices, num_tags) | tag_bitmap


@given(inputs=inputs1())
def test_log_likelihood(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    logits, mask, tag_indices = inputs

    num_tags = logits.size(2)
    crf = CRF(num_tags)
    crf2 = Crf(num_tags)
    crf2.transitions.data = crf.transitions.data

    actual = crf2(logits, mask).log_likelihood(tag_indices)
    expected = F.log_likelihood(crf(logits, mask), tag_indices, mask)

    torch.testing.assert_close(
        actual,
        expected,
    )

    actual.sum().neg().backward()
    expected.sum().neg().backward()  # type:ignore
    torch.testing.assert_close(crf.transitions.grad, crf2.transitions.grad)


@given(inputs=inputs1())
def test_argmax(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    logits, mask, _ = inputs
    logits.requires_grad_(False)

    num_tags = logits.size(2)
    label_set = LabelSet({f"LABEL-{i}" for i in range((num_tags - 1) // 4)})

    crf = CRF(num_tags)
    crf2 = Crf(num_tags)
    crf2.transitions.data = crf.transitions.data

    S = torch.tensor(label_set.start_states)
    E = torch.tensor(label_set.end_states)
    T = torch.tensor(label_set.transitions)

    torch.testing.assert_close(
        crf2(logits, mask, ~S, ~E, ~T).argmax,
        F.decode(F.constrain_log_potentials(crf(logits, mask), mask, S, E, T))[1],
    )


@given(inputs=inputs2())
def test_marginal_loglikelihood(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    logits, mask, tag_bitmap = inputs

    num_tags = logits.size(2)

    crf = CRF(num_tags)
    crf2 = Crf(num_tags)
    crf2.transitions.data = crf.transitions.data

    actual = crf2(logits, mask).marginal_log_likelihood(tag_bitmap)
    expected = F.marginal_log_likelihood(crf(logits, mask), tag_bitmap, mask)
    torch.testing.assert_close(actual, expected)

    actual.sum().neg().backward()
    expected.sum().neg().backward()  # type:ignore
    torch.testing.assert_close(crf.transitions.grad, crf2.transitions.grad)
