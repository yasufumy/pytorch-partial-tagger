from __future__ import annotations

import torch
from torch.nn import Module

from partial_tagger.crf.nn import CRF
from partial_tagger.decoders import ViterbiDecoder
from partial_tagger.encoders import BaseEncoder


class SequenceTagger(Module):
    """A sequence tagging model with a CRF layer.

    Args:
        encoder: An encoder module.
        decoder: A decoder module.

    Attributes:
        encoder: An encoder module.
        crf: A CRF layer.
        decoder: A decoder module.
    """

    def __init__(self, encoder: BaseEncoder, decoder: ViterbiDecoder):
        super(SequenceTagger, self).__init__()

        self.encoder = encoder
        self.crf = CRF(encoder.get_hidden_size())
        self.decoder = decoder

    def forward(
        self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log potentials and tag sequence.

        Args:
            inputs: An inputs representing input data feeding into the encoder module.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A pair of a [batch_size, sequence_length, num_tags, num_tags] float tensor
            and a [batch_size, sequence_length] integer tensor.
            The float tensor representing log potentials and
            the integer tensor representing tag sequence.
        """
        log_potentials = self.crf(self.encoder(inputs), mask)
        tag_indices = self.decoder(log_potentials, mask)
        return log_potentials, tag_indices

    def predict(
        self, inputs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """Predicts tag sequence from a given input.

        Args:
            inputs: An inputs representing input data feeding into the encoder module.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
             A [batch_size, sequence_length] integer tensor representing tag sequence.
        """
        return self(inputs, mask)[1]

    @property
    def padding_index(self) -> int:
        return self.decoder.padding_index
