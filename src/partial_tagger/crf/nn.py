from typing import Optional

import torch
from torch import nn

from . import NINF


class CRF(nn.Module):
    """A CRF layer.

    Args:
        num_tags: An integer representing the number of tags.

    """

    def __init__(self, num_tags: int) -> None:
        super(CRF, self).__init__()

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(
        self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes log potentials for a CRF.

        Args:
            logits: A [batch_size, sequence_length, num_tags] float tensor.
            mask: A [batch_size, sequence_length] boolean tensor.

        Returns:
            A [batch_size, sequence_length, num_tag, num_tags] float tensor
            representing log potentials.
        """
        if mask is None:
            mask = logits.new_ones(logits.shape[:-1], dtype=torch.bool)

        num_tags = logits.size(-1)
        initial_mask = torch.eye(num_tags, num_tags, device=logits.device).bool()

        # log potentials from the dummy initial token to the real initial token
        initial_log_potentials = logits[:, [0], :, None] * initial_mask + NINF * (
            ~initial_mask
        )
        log_potentials = torch.cat(
            (
                initial_log_potentials,
                logits[:, 1:, None, :] + self.transitions[None, None],
            ),
            dim=1,
        )

        mask_value = NINF * (~initial_mask)
        mask = mask[..., None, None]

        return log_potentials * mask + mask_value * (~mask)
