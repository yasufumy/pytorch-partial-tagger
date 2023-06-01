from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Span:
    start: int
    length: int

    def __hash__(self) -> int:
        return hash((self.start, self.length))


@dataclass(frozen=True, eq=True)
class Tag:
    span: Span
    label: str

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def length(self) -> int:
        return self.span.length


@dataclass(frozen=True, eq=True)
class CharBasedTags:
    tags: tuple[Tag, ...]
    text: str

    def __iter__(self) -> Iterator[Tag]:
        yield from self.tags

    def __repr__(self) -> str:
        tag_strs = []
        for tag in self:
            start = tag.start
            end = tag.start + tag.length
            tag_str = f"{self.text[start:end]} ({tag.label})"
            tag_strs.append(tag_str)
        return f"({', '.join(tag_strs)})"


class TokenizedText:
    def __init__(
        self,
        text: str,
        char_spans: tuple[Span | None, ...],
        token_indices: tuple[int, ...],
        addtional_token: str = "[Token]",
    ):
        self.__text = text
        self.__char_spans = char_spans
        self.__token_indices = token_indices
        self.__addtional_token = addtional_token

    def __repr__(self) -> str:
        text = []
        for span in self.__char_spans:
            if span is None:
                text.append(self.__addtional_token)
            else:
                start = span.start
                end = start + span.length
                text.append(self.__text[start:end])
        return " ".join(text)

    @property
    def num_tokens(self) -> int:
        return len(self.__char_spans)

    def get_text(self) -> str:
        return self.__text

    def get_token(self, token_index: int) -> str:
        char_span = self.__char_spans[token_index]
        if char_span is None:
            return self.__addtional_token
        else:
            start = char_span.start
            end = start + char_span.length
            return self.__text[start:end]

    def get_char_span(self, token_index: int) -> Span | None:
        return self.__char_spans[token_index]

    def convert_to_token_index(self, char_index: int) -> int:
        return self.__token_indices[char_index]

    def convert_to_char_span(self, token_span: Span) -> Span | None:
        char_span_start = self.__char_spans[token_span.start]
        char_span_end = self.__char_spans[token_span.start + token_span.length - 1]

        if char_span_start is None or char_span_end is None:
            return None

        return Span(
            char_span_start.start,
            char_span_end.start + char_span_end.length - char_span_start.start,
        )


@dataclass(frozen=True)
class SubwordBasedTags:
    tags: tuple[Tag, ...]
    tokenized_text: TokenizedText

    def __iter__(self) -> Iterator[Tag]:
        yield from self.tags

    def __repr__(self) -> str:
        tag_strs = []
        for tag in self:
            start = tag.start
            end = start + tag.length
            tag_str = " ".join(
                self.tokenized_text.get_token(token_index)
                for token_index in range(start, end)
            )
            tag_strs.append(f"{tag_str} ({tag.label})")
        return f"({', '.join(tag_strs)})"

    def get_char_based_tags(self) -> CharBasedTags:
        tags = []
        for tag in self.tags:
            char_span = self.tokenized_text.convert_to_char_span(tag.span)
            if char_span is not None:
                tags.append(Tag(char_span, tag.label))
        return CharBasedTags(tuple(tags), self.tokenized_text.get_text())


class LabelSet:
    def __init__(self, labels: set[str]):
        self.__labels = [*sorted(labels)]
        self.__status_kind = 4  # start, inside, end, unit

        self.__start_indices = [
            *range(
                1,
                self.get_tag_size(),
                self.__status_kind,
            )
        ]
        self.__unit_indices = [
            *range(
                self.__status_kind,
                self.get_tag_size(),
                self.__status_kind,
            )
        ]

        self.__label_ids = dict(zip(self.__labels, self.__start_indices))

    def get_outside_index(self) -> int:
        return 0

    def get_start_index(self, label: str) -> int:
        if label not in self.__label_ids:
            raise ValueError("Invalid label is given.")
        return self.__label_ids[label]

    def get_inside_index(self, label: str) -> int:
        return self.get_start_index(label) + 1

    def get_end_index(self, label: str) -> int:
        return self.get_start_index(label) + 2

    def get_unit_index(self, label: str) -> int:
        return self.get_start_index(label) + 3

    def get_label_size(self) -> int:
        return len(self.__labels)

    def get_tag_size(self) -> int:
        # (start, inside, end, unit) * label + outside status
        return self.__status_kind * self.get_label_size() + 1

    def get_label(self, index: int) -> str | None:
        if index < 0 or index >= self.get_tag_size():
            raise ValueError("Invalid index.")

        if index == self.get_outside_index():
            return None

        return self.__labels[bisect_left(self.__unit_indices, index)]

    def get_start_states(self) -> list[bool]:
        states = [False] * self.get_tag_size()
        # Always allowed starts from outside status
        states[self.get_outside_index()] = True

        for labeled_start_index in self.__start_indices:
            labeled_unit_index = labeled_start_index + 3
            states[labeled_start_index] = True
            states[labeled_unit_index] = True
        return states

    def get_end_states(self) -> list[bool]:
        states = [False] * self.get_tag_size()
        # Always allowed ends with outside status
        states[self.get_outside_index()] = True

        for labeled_unit_index in self.__unit_indices:
            labeled_end_index = labeled_unit_index - 1
            states[labeled_end_index] = True
            states[labeled_unit_index] = True
        return states

    def get_transitions(self) -> list[list[bool]]:
        transitions = [
            [False] * self.get_tag_size() for _ in range(self.get_tag_size())
        ]
        outside_index = self.get_outside_index()
        transitions[outside_index] = self.get_start_states()

        for labeled_start_index in self.__start_indices:
            labeled_inside_index = labeled_start_index + 1
            labeled_end_index = labeled_start_index + 2
            labeled_unit_index = labeled_start_index + 3

            transitions[labeled_start_index][labeled_inside_index] = True
            transitions[labeled_start_index][labeled_end_index] = True

            transitions[labeled_inside_index][labeled_inside_index] = True
            transitions[labeled_inside_index][labeled_end_index] = True

            transitions[labeled_end_index][outside_index] = True
            for i in self.__start_indices + self.__unit_indices:
                transitions[labeled_end_index][i] = True

            transitions[labeled_unit_index][outside_index] = True
            for i in self.__start_indices + self.__unit_indices:
                transitions[labeled_unit_index][i] = True

        return transitions
