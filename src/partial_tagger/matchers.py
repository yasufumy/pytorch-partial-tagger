from __future__ import annotations

from abc import ABCMeta, abstractmethod

from spacy import Language

from partial_tagger.data import Span, Tag


class BaseMatcher(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, text: str) -> set[Tag]:
        raise NotImplementedError


class SpacyMatcher(BaseMatcher):
    def __init__(self, nlp: Language):
        if not nlp.has_pipe("entity_ruler"):
            raise ValueError("Please setup your EntityRuler.")

        self.__nlp = nlp

    def __call__(self, text: str) -> set[Tag]:
        doc = self.__nlp(text)
        tags = []
        for ent in doc.ents:
            tags.append(Tag(Span(ent.start_char, len(ent.text)), ent.label_))
        return set(tags)
