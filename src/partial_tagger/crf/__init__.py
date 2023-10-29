import torch

from partial_tagger.crf.core import BaseCrfDistribution, Crf

# Negative infinity
NINF = torch.finfo(torch.float16).min


__all__ = ["BaseCrfDistribution", "Crf", "NINF"]
