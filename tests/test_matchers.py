from typing import TYPE_CHECKING

import pytest
import spacy

from partial_tagger.data.core import Span, Tag
from partial_tagger.matchers import SpacyMatcher

if TYPE_CHECKING:
    from spacy.pipeline import EntityRuler


@pytest.fixture()
def matcher() -> SpacyMatcher:
    nlp = spacy.blank("en")
    ruler: EntityRuler = nlp.add_pipe("entity_ruler")  # type:ignore
    ruler.add_patterns(
        [{"label": "LOC", "pattern": "Tokyo"}, {"label": "LOC", "pattern": "Japan"}]
    )
    return SpacyMatcher(nlp)


def test_matches_valid_tags(matcher: SpacyMatcher) -> None:
    text = "Tokyo is the capital of Japan."
    expected = {Tag(Span(0, 5), "LOC"), Tag(Span(24, 5), "LOC")}

    tags = matcher(text)

    assert expected == tags
