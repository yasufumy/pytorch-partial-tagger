from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto


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


class TokenizedText:
    """A TokenizedText represents a text that has been tokenized and provides
    convenient methods to access its tokens and spans.

    Args:
        text: An original text.
        char_spans: A tuple of character spans for each token, or None if
            there is no corresponding span.
        token_indices: A tuple of token indices for each character, or -1 if there is
            no corresponding token.
        addtional_token: A string used to represent token without no corresponding
            character span. Defaults to "[Token]".
    """

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
        """Converts a character index to its corresponding token index.

        Args:
            char_index: A character index.

        Returns:
            A corresponding token index, or -1 if there is no corresponding token.

        """
        return self.__token_indices[char_index]

    def convert_to_char_span(self, token_span: Span) -> Span | None:
        """Converts a token span to its corresponding character span.

        Args:
            token_span: An instance of Span corresponding a token.

        Returns:
            A corresponding character span, or None if there is no corresponding
            character span.
        """
        char_span_start = self.__char_spans[token_span.start]
        char_span_end = self.__char_spans[token_span.start + token_span.length - 1]

        if char_span_start is None or char_span_end is None:
            return None

        return Span(
            char_span_start.start,
            char_span_end.start + char_span_end.length - char_span_start.start,
        )


@dataclass(frozen=True, eq=True)
class CharBasedTags:
    """Character-based tags represent a collection of tags that occur in a text
    on the character level.

    Args:
        tags: A tuple of instances of Tag representing tags that occur in a text.
        text: A text in which tags are defined.

    """

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

    def convert_to_token_based(self, tokenized_text: TokenizedText) -> "TokenBasedTags":
        """Converts an instance CharBasedTags to an instance of TokenBasedTags
        based on a provided instance of TokenizedText. Note that this operation
        is irreversible. For example, if the given tokenized text is truncated,
        tags associated with a truncated part will be ignored.

        Args:
            tokenized_text: An instance of TokenizedText.

        Returns:
            An instance of TokenBasedTags.

        """
        if self.text != tokenized_text.get_text():
            raise ValueError("The text doesn't match")

        tags = []
        for tag in self.tags:
            start = tokenized_text.convert_to_token_index(tag.start)
            end = tokenized_text.convert_to_token_index(tag.start + tag.length - 1)
            if start == -1 or end == -1:
                # There is no char span which strictly corresponds a given tag.
                continue
            length = end - start + 1
            tags.append(Tag(Span(start, length), tag.label))

        return TokenBasedTags(tuple(tags), tokenized_text)


@dataclass(frozen=True)
class TokenBasedTags:
    """Token-based tags represent a collection of tags that occur in a text
    on the token level.

    Args:
        tags: A tuple of instances of Tag representing tags that occur in a text.
        text: A tokenized text in which tags are defined.

    """

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

    @property
    def num_tokens(self) -> int:
        return self.tokenized_text.num_tokens

    def convert_to_char_based(self) -> CharBasedTags:
        """Converts an instance of TokenBasedTags to an instance of CharBasedTags.

        Returns:
            An instance of CharBasedTags.
        """
        tags = []
        for tag in self.tags:
            char_span = self.tokenized_text.convert_to_char_span(tag.span)
            if char_span is not None:
                tags.append(Tag(char_span, tag.label))
        return CharBasedTags(tuple(tags), self.tokenized_text.get_text())

    def get_tag_indices(
        self, label_set: LabelSet, unknown_index: int = -100
    ) -> list[int]:
        """Returns a list of active tag indices.

        Args:
            label_set: An instance of LabelSet.
            unknown_index: An integer representing an index for an unknown tag.
                Defaults to -100.

        Returns:
            A list of integers, where each integer represents an active tag.

        """
        tag_indices = [unknown_index] * self.tokenized_text.num_tokens

        for token_index in range(self.tokenized_text.num_tokens):
            span = self.tokenized_text.get_char_span(token_index)
            if span is None:
                tag_indices[token_index] = label_set.get_outside_index()

        for tag in self.tags:
            start = tag.start
            end = tag.start + tag.length - 1
            if start == end:
                tag_indices[start] = label_set.get_unit_index(tag.label)
            else:
                tag_indices[start] = label_set.get_start_index(tag.label)
                tag_indices[start + 1 : end] = [
                    label_set.get_inside_index(tag.label)
                ] * (end - start - 1)
                tag_indices[end] = label_set.get_end_index(tag.label)

        return tag_indices

    def get_tag_bitmap(self, label_set: LabelSet) -> list[list[bool]]:
        """Returns a tag bitmap indicating the presence of active tags for each token.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            A list of lists of booleans, where each boolean represents an active tag.

        """
        tag_bitmap = [
            [False] * label_set.get_tag_size() for _ in range(self.num_tokens)
        ]
        for token_index in range(self.num_tokens):
            span = self.tokenized_text.get_char_span(token_index)
            if span is None:
                tag_bitmap[token_index][label_set.get_outside_index()] = True

        for tag in sorted(
            self,
            key=lambda tag: (tag.start, tag.start + tag.length),
        ):
            start = tag.start
            end = tag.start + tag.length - 1  # inclusive
            if start == end:
                tag_bitmap[start][label_set.get_unit_index(tag.label)] = True
            else:
                tag_bitmap[start][label_set.get_start_index(tag.label)] = True
                for i in range(start + 1, end):
                    tag_bitmap[i][label_set.get_inside_index(tag.label)] = True
                tag_bitmap[end][label_set.get_end_index(tag.label)] = True

        for bit in tag_bitmap:
            if sum(bit) == 0:
                bit[:] = [True] * label_set.get_tag_size()

        return tag_bitmap


class Status(Enum):
    START = auto()
    INSIDE = auto()
    END = auto()
    UNIT = auto()


class LabelSet:
    """A label set represents a set of labels used for tagging, where each label
    has four states (start, inside, end, unit).

    Args:
        labels: A set of strings, where each string represents a label.
    """

    def __init__(self, labels: set[str]):
        self.__labels = [*sorted(labels)]
        self.__status_size = len(Status)  # start, inside, end, unit

        self.__start_indices = [
            *range(
                1,
                self.get_tag_size(),
                self.__status_size,
            )
        ]
        self.__unit_indices = [
            *range(
                self.__status_size,
                self.get_tag_size(),
                self.__status_size,
            )
        ]

        self.__label_ids = dict(zip(self.__labels, self.__start_indices))

    def __contains__(self, label: str) -> bool:
        i = bisect_left(self.__labels, label)
        if i >= len(self.__labels):
            return False
        return self.__labels[i] == label

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

    def get_labels(self) -> list:
        return self.__labels

    def get_label_size(self) -> int:
        return len(self.__labels)

    def get_tag_size(self) -> int:
        # (start, inside, end, unit) * label + outside status
        return self.__status_size * self.get_label_size() + 1

    def get_label(self, index: int) -> str | None:
        """Returns a label associated with a given index.

        Args:
            index: An integer representing a label with a state.

        Returns:
            A string representing a label, or None if a given index represents
            an outside status.
        """
        if index < 0 or index >= self.get_tag_size():
            raise ValueError("Invalid index.")

        if index == self.get_outside_index():
            return None

        return self.__labels[bisect_left(self.__unit_indices, index)]

    def get_status(self, index: int) -> Status | None:
        """Returns a status associated with a given index.

        Args:
            index: An integer representing a label with a state.

        Returns:
            A status associated with an index, or None if an given index represents
            an outside status.

        """
        if index < 0 or index >= self.get_tag_size():
            raise ValueError("Invalid index.")

        label = self.get_label(index)
        if label is None:
            return None
        elif self.get_start_index(label) == index:
            return Status.START
        elif self.get_inside_index(label) == index:
            return Status.INSIDE
        elif self.get_end_index(label) == index:
            return Status.END
        elif self.get_unit_index(label) == index:
            return Status.UNIT
        else:
            raise ValueError("Invalid index.")

    def get_start_states(self) -> list[bool]:
        """Returns a list of booleans representing an allowed start states.

        Returns:
            A list of booleans representing allowed start states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.get_tag_size()
        # Always allowed starts from outside status
        states[self.get_outside_index()] = True

        for labeled_start_index in self.__start_indices:
            labeled_unit_index = labeled_start_index + 3
            states[labeled_start_index] = True
            states[labeled_unit_index] = True
        return states

    def get_end_states(self) -> list[bool]:
        """Returns a list of booleans representing an allowed end states.

        Returns:
            A list of booleans representing allowed end states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.get_tag_size()
        # Always allowed ends with outside status
        states[self.get_outside_index()] = True

        for labeled_unit_index in self.__unit_indices:
            labeled_end_index = labeled_unit_index - 1
            states[labeled_end_index] = True
            states[labeled_unit_index] = True
        return states

    def get_transitions(self) -> list[list[bool]]:
        """Returns a list of lists of booleans representing
        allowed transitions between tags.

        Returns:
            A list of lists of booleans representing allowed transitions between tags,
            where each item is: True for an allowed transition and False otherwise.

        """
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
