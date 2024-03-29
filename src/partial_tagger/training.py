from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Tuple, cast

import torch
from sequence_label import LabelSet, SequenceLabel
from sequence_label.core import Base
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.data import DataLoader

from partial_tagger.data.collators import TrainingCollator
from partial_tagger.metric import Metric
from partial_tagger.recognizer import Recognizer
from partial_tagger.tagger import SequenceTagger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger

    from sequence_classifier.crf import BaseCrfDistribution
    from sequence_label import LabelAlignment

    from partial_tagger.data.collators import BaseCollator, Batch
    from partial_tagger.encoders.base import BaseEncoderFactory


class SlantedTriangular:
    def __init__(self, max_steps: int, cut_frac: float = 0.1, ratio: int = 16):
        self.__cut_frac = cut_frac
        self.__cut = int(max_steps * cut_frac)
        self.__ratio = ratio

    def __call__(self, step: int) -> float:
        if step < self.__cut:
            p = step / self.__cut
        else:
            p = 1 - (step - self.__cut) / (self.__cut * (1 / self.__cut_frac - 1))
        return (1 + p * (self.__ratio - 1)) / self.__ratio


def compute_partially_supervised_loss(
    crf_distribution: BaseCrfDistribution,
    tag_bitmap: torch.Tensor,
    outside_index: int,
    target_entity_ratio: float = 0.15,
    entity_ratio_margin: float = 0.05,
    balancing_coefficient: int = 10,
) -> torch.Tensor:
    """Computes the loss proposed in Effland and Collins. '21.

    Args:
        log_potentials: A [batch_size, sequence_length, num_tag, num_tag] float tensor
            representing log potentials.
        tag_bitmap: A [batch_size, sequence_length, num_tag] boolean tensor indicating
            all active tags at each index.
        outside_index: An integer representing a non-entity index.
        target_entity_ratio: A float representing a target entity ratio
            for training. Defaults to 0.15.
        entity_ratio_margin: A float representing a margin for the entity ratio.
            Defaults to 0.05.
        balancing_coefficient: An integer representing a balancing coefficient
            for the loss function. Defaults to 10.

    Returns:
        A float representing loss.
    """
    with torch.enable_grad():  # type:ignore
        # log partition
        log_partitions = crf_distribution.log_partitions

        # marginal probabilities
        p = cast(torch.Tensor, log_partitions.marginals)

    expected_entity_count = (
        p[:, :, :outside_index].sum() + p[:, :, outside_index + 1 :].sum()
    )
    expected_entity_ratio = expected_entity_count / p.sum()
    expected_entity_ratio_loss = torch.clamp(
        (expected_entity_ratio - target_entity_ratio).abs() - entity_ratio_margin,
        min=0,
    )

    score = crf_distribution.log_multitag_scores(tag_bitmap=tag_bitmap)
    supervised_loss = (log_partitions.value - score).mean()

    return cast(
        torch.Tensor,
        supervised_loss + balancing_coefficient * expected_entity_ratio_loss,
    )


def create_tag_bitmap(
    label_set: LabelSet,
    labels: tuple[SequenceLabel, ...],
    alignments: tuple[LabelAlignment, ...],
    device: torch.device,
) -> torch.Tensor:
    tag_bitmap = label_set.encode_to_tag_bitmap(labels=labels, alignments=alignments)
    for bitmap, alignment in zip(tag_bitmap, alignments):
        for i, length in enumerate(alignment.get_span_lengths(Base.SOURCE)):
            if length == 0 or not bitmap[i][label_set.outside_index]:
                continue
            bitmap[i][:] = [True] * label_set.state_size
    return torch.tensor(tag_bitmap, device=device)


class Trainer:
    """A trainer for fitting the parameters of a tagger based on a given dataset.

    Args:
        collator: Any instance of the classes that inherit BaseCollator.
        encoder_factory: An encoder factory for creating encoders.
    """

    def __init__(
        self,
        collator: BaseCollator,
        encoder_factory: BaseEncoderFactory,
    ):
        self.__collator = collator
        self.__encoder_factory = encoder_factory

    def __call__(
        self,
        train_dataset: list[tuple[str, SequenceLabel]],
        validation_dataset: list[tuple[str, SequenceLabel]],
        device: torch.device,
        batch_size: int = 15,
        num_epochs: int = 20,
        learning_rate: float = 2e-5,
        gradient_clip_value: float = 5.0,
        target_entity_ratio: float = 0.15,
        entity_ratio_margin: float = 0.05,
        balancing_coefficient: int = 10,
        padding_index: int = -1,
        logger: Logger | None = None,
    ) -> Recognizer:
        """Trains an instance of SequenceTagger.

        Args:
            train_dataset: A list of training data tuples containing text and tags.
            validation_dataset: A list of validation data tuples
                containing text and tags.
            batch_size: An integer representing a batch size for training.
                Defaults to 15.
            num_epochs: An integer representing the number of epochs for training.
                Defaults to 20.
            learning_rate: A float representing a learning rate for optimization.
                Defaults to 2e-5.
            gradient_clip_value: A float representing a maximum gradient value
                for clipping. Defaults to 5.0.
            target_entity_ratio: A float representing a target entity ratio
                for training. Defaults to 0.15.
            entity_ratio_margin: A float representing a margin for the entity ratio.
                Defaults to 0.05.
            balancing_coefficient: An integer representing a balancing coefficient
                for the loss function. Defaults to 10.
            padding_index: An integer representing an index for padding. Defaults to -1.
            device: A device to be used for training.
            logger: A logger for logging training progress. Defaults to None.

        Returns:
            An instance of Recognizer which predicts character-based tags
            from a given text.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())

        # Create a label_set
        label_set = LabelSet(
            labels={tag.label for _, label in train_dataset for tag in label.tags},
            padding_index=padding_index,
        )

        tagger = SequenceTagger(
            encoder=self.__encoder_factory.create(label_set),
            padding_index=label_set.padding_index,
            start_states=label_set.start_states,
            end_states=label_set.end_states,
            transitions=label_set.transitions,
        )
        tagger.to(device)

        collator = TrainingCollator(self.__collator)

        train_dataloader: Sequence[
            tuple[Batch, tuple[LabelAlignment, ...], tuple[SequenceLabel, ...]]
        ] = DataLoader[Tuple[str, SequenceLabel]](
            train_dataset,  # type:ignore
            collate_fn=collator,
            batch_size=batch_size,
        )
        validation_dataloader: Sequence[
            tuple[Batch, tuple[LabelAlignment, ...], tuple[SequenceLabel, ...]]
        ] = DataLoader[Tuple[str, SequenceLabel]](
            validation_dataset,  # type:ignore
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            tagger.parameters(), lr=learning_rate, weight_decay=0.0
        )
        schedular = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            SlantedTriangular(len(train_dataloader) * num_epochs),
        )
        best_f1_score = float("-inf")
        best_tagger_state = io.BytesIO()

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            tagger.train()

            for batch, alignments, labels in train_dataloader:
                batch = batch.to(device)

                optimizer.zero_grad()

                dist = tagger(batch.tagger_inputs, batch.mask)

                loss = compute_partially_supervised_loss(
                    crf_distribution=dist,
                    tag_bitmap=create_tag_bitmap(
                        label_set=label_set,
                        labels=labels,
                        alignments=alignments,
                        device=device,
                    ),
                    outside_index=label_set.outside_index,
                    target_entity_ratio=target_entity_ratio,
                    entity_ratio_margin=entity_ratio_margin,
                    balancing_coefficient=balancing_coefficient,
                )
                loss.backward()  # type:ignore

                clip_grad_value_(tagger.parameters(), clip_value=gradient_clip_value)

                optimizer.step()
                schedular.step()

                epoch_loss += loss.item() * len(alignments)

            tagger.eval()
            metric = Metric()
            for batch, alignments, labels in validation_dataloader:
                batch = batch.to(device)

                tag_indices = tagger.predict(batch.tagger_inputs, batch.mask)

                predictions = label_set.decode(
                    tag_indices=tag_indices.tolist(), alignments=alignments
                )

                metric(predictions, labels)

            scores = metric.get_scores()

            if best_f1_score < scores["micro_f1_score"]:
                best_f1_score = scores["micro_f1_score"]
                best_tagger_state.truncate(0)
                best_tagger_state.seek(0)
                torch.save(obj=tagger.state_dict(), f=best_tagger_state)

            logger.info(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    **{f"validation_{key}": value for key, value in scores.items()},
                }
            )

        best_tagger_state.seek(0)
        tagger.load_state_dict(torch.load(best_tagger_state))

        return Recognizer(tagger=tagger, collator=self.__collator, label_set=label_set)
