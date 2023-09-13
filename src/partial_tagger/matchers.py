from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, cast

from sequence_label import SequenceLabel
from sequence_label.core import TagDict

if TYPE_CHECKING:
    from spacy.language import Language


class BaseMatcher(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, text: str) -> SequenceLabel:
        raise NotImplementedError


class SpacyMatcher(BaseMatcher):
    def __init__(self, nlp: Language):
        if not nlp.has_pipe("entity_ruler"):
            raise ValueError("Please setup your EntityRuler.")

        self.__nlp = nlp

    def __call__(self, text: str) -> SequenceLabel:
        doc = self.__nlp(text)
        tags = []
        for ent in doc.ents:
            tags.append(
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            )
        return SequenceLabel.from_dict(tags=cast(List[TagDict], tags), size=len(text))
