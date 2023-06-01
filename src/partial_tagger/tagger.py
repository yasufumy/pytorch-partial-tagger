from __future__ import annotations

import torch
from torch.nn import Module

from .crf.nn import CRF
from .data.batch import TaggerInputs
from .decoders import ViterbiDecoder
from .embedders import BaseEmbedder
from .encoders import BaseEncoder


class SequenceTagger(Module):
    """Sequence tagger.

    Args:
        embedder: An embedder.
        encoder: An encoder.
        decoder: A decoder.
    """

    def __init__(
        self, embedder: BaseEmbedder, encoder: BaseEncoder, decoder: ViterbiDecoder
    ):
        super(SequenceTagger, self).__init__()

        self.embedder = embedder
        self.encoder = encoder
        self.crf = CRF(encoder.get_hidden_size())
        self.decoder = decoder

    def forward(
        self, inputs: TaggerInputs, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log potentials and tag sequence.

        Args:
            inputs: An inputs representing input data feeding into an embedder.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A pair of a [batch_size, sequence_length, num_tags, num_tags] float tensor
            and a [batch_size, sequence_length] integer tensor.
            The float tensor representing log potentials and
            the integer tensor representing tag sequence.
        """
        embeddings = self.embedder(inputs)
        log_potentials = self.crf(self.encoder(embeddings, mask), mask)
        tag_indices = self.decoder(log_potentials, mask)
        return log_potentials, tag_indices

    def predict(self, inputs: TaggerInputs, mask: torch.Tensor) -> torch.Tensor:
        return self(inputs, mask)[1]
