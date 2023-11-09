import torch

# Negative infinity
NINF = torch.finfo(torch.float16).min


__all__ = ["NINF"]
