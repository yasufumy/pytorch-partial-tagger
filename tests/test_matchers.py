from typing import TYPE_CHECKING

import pytest
import spacy
from sequence_label import SequenceLabel

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
    expected = SequenceLabel.from_dict(
        [
            {"start": 0, "end": 5, "label": "LOC"},
            {"start": 24, "end": 29, "label": "LOC"},
        ],
        size=len(text),
    )

    tags = matcher(text)

    assert expected == tags
