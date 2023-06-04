from typing import List, Tuple

from .core import (  # NOQA
    CharBasedTags,
    LabelSet,
    Span,
    Tag,
    TokenBasedTags,
    TokenizedText,
)

Dataset = List[Tuple[str, CharBasedTags]]
