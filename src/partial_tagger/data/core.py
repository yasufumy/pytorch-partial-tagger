from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class Span:
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be zero or positive: {self.start}")

        if self.length <= 0:
            raise ValueError(f"length must be positive: {self.length}")


@dataclass(frozen=True)
class Tag:
    span: Span
    label: str

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def length(self) -> int:
        return self.span.length

    @classmethod
    def create(cls, start: int, end: int, label: str) -> Tag:
        """Creates an instance of Tag.

        Args:
            start: An integer representing a position in text where a tag starts.
            end: An integer representing a position in text where a tag ends.
                Note that an end is expected to be exclusive.
            label: A string representing what you want to assign to a span in a text.

        Returns:
            An instance of Tag.
        """
        if start < 0:
            raise ValueError(f"start must be zero or positive: {start}")

        if start >= end:
            raise ValueError(f"end must be greater than start: {start} >= {end}")

        length = end - start
        return cls(Span(start, length), label)


class Alignment:
    """An alignment class responsible for manipulating character-based tags based on
    a tokenization result, which is useful for encoding tags to tag_indices/tag_bitmap
    and decoding tag_indices/tag_bitmap to tags.

    Args:
        char_spans: A tuple of character spans for each token, or None if
            there is no corresponding span.
        token_indices: A tuple of token indices for each character, or -1 if there is
            no corresponding token.

    Attributes:
        char_length: The text length before tokenization.
        num_tokens: The number of tokens after tokenization.
    """

    def __init__(
        self, char_spans: tuple[Span | None, ...], token_indices: tuple[int, ...]
    ):
        num_tokens = len(char_spans)
        if not all(index == -1 or 0 <= index < num_tokens for index in token_indices):
            raise ValueError(
                "Each item in token_indices must be -1 or"
                f" in between 0 and {num_tokens - 1}: {token_indices}"
            )

        char_length = len(token_indices)
        if not all(
            0 <= span.start < char_length  # check if start is valid
            and 0 <= span.start + span.length - 1 < char_length  # check if end is valid
            for span in char_spans
            if span is not None
        ):
            raise ValueError(
                "Each span in char_spans must be None or"
                f" in between 0 and {char_length - 1}: {char_spans}"
            )

        self.__char_spans = char_spans
        self.__token_indices = token_indices

    @property
    def char_length(self) -> int:
        return len(self.__token_indices)

    @property
    def num_tokens(self) -> int:
        return len(self.__char_spans)

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
            start=char_span_start.start,
            length=char_span_end.start + char_span_end.length - char_span_start.start,
        )

    def convert_to_char_based(self, tags: set[Tag]) -> set[Tag]:
        """Converts token-based tags to character-based tags.

        Args:
            tags: A set of token-based tags

        Returns:
            A set of character-based tags.
        """
        for tag in tags:
            if tag.start < 0 or tag.start >= self.num_tokens:
                raise ValueError(
                    "An invalid tag is found. start must be"
                    f" in between 0 and {self.num_tokens - 1}: {tag.start}"
                )
            end = tag.start + tag.length
            if end < 0 or end > self.num_tokens:
                raise ValueError(
                    "An invalid tag is found. length must be"
                    f" in between 1 and {self.num_tokens}: {tag.length}"
                )

        aligned_tags = []
        for tag in tags:
            char_span = self.convert_to_char_span(tag.span)
            if char_span is not None:
                aligned_tags.append(Tag(char_span, tag.label))

        return set(aligned_tags)

    def convert_to_token_based(self, tags: set[Tag]) -> set[Tag]:
        """Converts character-based tags to token-based tags. Note that this operation
        is irreversible. For example, if a text is truncated in tokenization,
        tags associated with a truncated part will be ignored.

        Args:
            tags: A set of character-based tags

        Returns:
            A set of token-based tags.
        """
        for tag in tags:
            if tag.start < 0 or tag.start >= self.char_length:
                raise ValueError(
                    "An invalid tag is found. start must be"
                    f" in between 0 and {self.char_length - 1}: {tag.start}"
                )
            end = tag.start + tag.length
            if end < 0 or end > self.char_length:
                raise ValueError(
                    "An invalid tag is found. length must be"
                    f" in between 1 and {self.char_length}: {tag.length}"
                )

        aligned_tags = []
        for tag in tags:
            start = self.__token_indices[tag.start]
            end = self.__token_indices[tag.start + tag.length - 1]
            if start == -1 or end == -1:
                # There is no char span which strictly corresponds a given tag.
                continue
            length = end - start + 1
            aligned_tags.append(Tag(Span(start, length), tag.label))

        return set(aligned_tags)

    def create_char_based_tags(
        self, tag_indices: list[int], label_set: LabelSet
    ) -> set[Tag]:
        """Creates a set of character-based tags from given tag indices.

        Args:
            tag_indices: A list of integer, where each item represents a tag index.
            label_set: An instance of LabelSet.

        Returns:
            A set of character-based tags.
        """
        if self.num_tokens != len(tag_indices):
            raise ValueError(
                f"The length of tag_indices ({len(tag_indices)}) must be equal"
                f" to the number of tokens ({self.num_tokens})"
            )

        if not all(0 <= index < label_set.get_tag_size() for index in tag_indices):
            raise ValueError(
                "Each index in tag_indices must be in between"
                f" 0 and {label_set.get_tag_size()}"
            )

        tags = []
        stack: list[str] = []
        for pos, index in enumerate(tag_indices):
            status = label_set.get_status(index)
            label = label_set.get_label(index)
            if status is None or label is None:
                continue

            if status == Status.UNIT:
                tags.append(Tag(Span(pos, 1), label))
            elif status == Status.END:
                if stack and stack[-1] == label:
                    length = len(stack)
                    tags.append(Tag(Span(pos - length, length + 1), label))
                stack.clear()
            elif status == Status.START or status == Status.INSIDE:
                if not stack or stack[-1] == label:
                    stack.append(label)
                else:
                    stack.clear()
            else:
                raise ValueError("Invalid status.")

        return self.convert_to_char_based(set(tags))

    def create_tag_indices(
        self, tags: set[Tag], label_set: LabelSet, unknown_index: int = -100
    ) -> list[int]:
        """Creates a list of active tag indices where given tags are expected
        to be character-based.

        Args:
            label_set: An instance of LabelSet.
            unknown_index: An integer representing an index for an unknown tag.
                Defaults to -100.

        Returns:
            A list of integers, where each integer represents an active tag.

        """
        tags = self.convert_to_token_based(tags)

        tag_indices = [unknown_index] * self.num_tokens

        for token_index in range(self.num_tokens):
            span = self.__char_spans[token_index]
            if span is None:
                tag_indices[token_index] = label_set.get_outside_index()

        for tag in sorted(
            tags,
            key=lambda tag: (tag.start, tag.start + tag.length),
        ):
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

    def create_tag_bitmap(
        self, tags: set[Tag], label_set: LabelSet
    ) -> list[list[bool]]:
        """Creates a tag bitmap indicating the presence of active tags for each token
        where given tags are expected to be character-based.

        Args:
            label_set: An instance of LabelSet.

        Returns:
            A list of lists of booleans, where each boolean represents an active tag.

        """
        tags = self.convert_to_token_based(tags)

        tag_bitmap = [
            [False] * label_set.get_tag_size() for _ in range(self.num_tokens)
        ]
        for token_index in range(self.num_tokens):
            span = self.__char_spans[token_index]
            if span is None:
                tag_bitmap[token_index][label_set.get_outside_index()] = True

        for tag in sorted(
            tags,
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


@dataclass(frozen=True)
class Alignments:
    """A batched version of the Alignment class.

    Args:
        alignments: A tuple of an instance of Alignment.

    Attributes:
        alignments: A tuple of an instance of Alignment.
    """

    alignments: tuple[Alignment, ...]

    def __len__(self) -> int:
        return len(self.alignments)

    def get_tag_indices(
        self,
        tags_batch: tuple[set[Tag], ...],
        label_set: LabelSet,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> list[list[int]]:
        """Encodes a batch of character-based tags into tag indices.

        Args:
            tags_batch: A tuple of character-based tags.
            label_set: An instance of LabelSet.
            padding_index: An integer representing an index to pad a tensor.
                Defaults to -1.
            unknown_index: An integer representing an index for an unknown tag.
                Defaults to -100.

        Returns:
            A 2D integer list of tag indices whose size is
            [batch_size, sequence_length].
        """
        max_length = max(alignment.num_tokens for alignment in self.alignments)

        tag_indices = []

        for tags, alignment in zip(tags_batch, self.alignments):
            indices = alignment.create_tag_indices(
                tags=tags, label_set=label_set, unknown_index=unknown_index
            )
            tag_indices.append(indices + [padding_index] * (max_length - len(indices)))

        return tag_indices

    def get_tag_bitmap(
        self, tags_batch: tuple[set[Tag], ...], label_set: LabelSet
    ) -> list[list[list[bool]]]:
        """Encodes a batch of character-based tags into tag bitmap.

        Args:
            tags_batch: A tuple of character-based tags.
            label_set: An instance of LabelSet.

        Returns:
            A 3D boolean list of tag bitmap whose size is
            [batch_size, sequence_length, num_tags].
        """
        max_length = max(alignment.num_tokens for alignment in self.alignments)

        tag_bitmap = []

        for tags, alignment in zip(tags_batch, self.alignments):
            bitmap = alignment.create_tag_bitmap(tags=tags, label_set=label_set)
            tag_bitmap.append(
                bitmap
                + [
                    [False] * label_set.get_tag_size()
                    for _ in range(max_length - len(bitmap))
                ]
            )

        return tag_bitmap

    def create_char_based_tags(
        self, tag_indices: list[list[int]], label_set: LabelSet, padding_index: int = -1
    ) -> tuple[set[Tag], ...]:
        """Creates character-based tags from given tag indices
        based on alignment information.

        Args:
            tag_indices: A [batch_size, sequence_length] integer tensor of tag indices.
            label_set: An instance of LabelSet to use for tag conversion.
            padding_index: An integer representing a padding index. Defaults to -1.

        Returns:
            A tuple where each item is a set of character-based tags.
        """
        tag_indices = [[i for i in x if i != padding_index] for x in tag_indices]

        if len(tag_indices) != len(self.alignments):
            raise ValueError(
                f"Batch size mismatch: {len(tag_indices)} != {len(self.alignments)}"
            )

        tags_batch = []

        for alignment, indices in zip(self.alignments, tag_indices):
            tags_batch.append(
                alignment.create_char_based_tags(
                    tag_indices=indices, label_set=label_set
                )
            )

        return tuple(tags_batch)

    @classmethod
    def from_offset_mapping(
        cls, offset_mapping: list[list[tuple[int, int]]], char_lengths: tuple[int, ...]
    ) -> Alignments:
        """Creates an instance of Alignments from Hugging Face offset_mapping.

        Args:
            offset_mapping: A list of list of a pair of integers where each pair
                represents a character span that corresponds to a token.
            char_lengths: A tuple of integers where each item represents the length
            of the original text before tokenization.

        Returns:
            An instance of Alignments.
        """
        alignments = []
        for mapping, char_length in zip(offset_mapping, char_lengths):
            char_spans = tuple(
                Span(start, end - start) if start != end else None
                for start, end in mapping
            )
            token_indices = [-1] * char_length
            for token_index, char_span in enumerate(char_spans):
                if char_span is None:
                    continue
                start = char_span.start
                end = char_span.start + char_span.length
                token_indices[start:end] = [token_index] * char_span.length

            alignments.append(
                Alignment(char_spans=char_spans, token_indices=tuple(token_indices))
            )
        return cls(alignments=tuple(alignments))


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
