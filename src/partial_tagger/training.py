from __future__ import annotations

import io
import logging
from logging import Logger
from typing import cast

import torch
from torch.utils.data import DataLoader

from .crf import functional as F
from .data import Dataset, LabelSet
from .data.batch import Collator
from .data.batch.tag import TagsBatch
from .data.batch.text import BaseTokenizer, TextBatch
from .decoders.viterbi import Contrainer, ViterbiDecoder
from .encoders import BaseEncoderFactory
from .metric import Metric
from .recognizer import Recognizer
from .tagger import SequenceTagger


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


def expected_entity_ratio_loss(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: torch.Tensor,
    outside_index: int,
    target_entity_ratio: float = 0.15,
    entity_ratio_margin: float = 0.05,
    balancing_coefficient: int = 10,
) -> torch.Tensor:
    with torch.enable_grad():
        # log partition
        log_Z = F.forward_algorithm(log_potentials)

        # marginal probabilities
        p = torch.autograd.grad(log_Z.sum(), log_potentials, create_graph=True)[0].sum(
            dim=-1
        )
    p *= mask[..., None]

    expected_entity_count = (
        p[:, :, :outside_index].sum() + p[:, :, outside_index + 1 :].sum()
    )
    expected_entity_ratio = expected_entity_count / p.sum()
    expected_entity_ratio_loss = torch.clamp(
        (expected_entity_ratio - target_entity_ratio).abs() - entity_ratio_margin,
        min=0,
    )

    score = F.multitag_sequence_score(log_potentials, tag_bitmap, mask)
    supervised_loss = (log_Z - score).mean()

    return supervised_loss + balancing_coefficient * expected_entity_ratio_loss


class Trainer:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        encoder_factory: BaseEncoderFactory,
        batch_size: int = 15,
        num_epochs: int = 20,
        learning_rate: float = 2e-5,
        gradient_clip_value: float = 5.0,
        target_entity_ratio: float = 0.15,
        entity_ratio_margin: float = 0.05,
        balancing_coefficient: int = 10,
        padding_index: int = -1,
    ):
        self.__tokenizer = tokenizer
        self.__encoder_factory = encoder_factory
        self.__batch_size = batch_size
        self.__num_epochs = num_epochs
        self.__learning_rate = learning_rate
        self.__gradient_clip_value = gradient_clip_value
        self.__target_entity_ratio = target_entity_ratio
        self.__entity_ratio_margin = entity_ratio_margin
        self.__balancing_coefficient = balancing_coefficient
        self.__padding_index = padding_index

    def __call__(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        device: torch.device,
        logger: Logger | None = None,
    ) -> Recognizer:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())

        # Create a label_set
        labels = set()
        for _, tags in train_dataset:
            for tag in tags:
                if tag.label not in labels:
                    labels.add(tag.label)
        label_set = LabelSet(labels)

        tagger = SequenceTagger(
            self.__encoder_factory.create(label_set),
            ViterbiDecoder(
                self.__padding_index,
                Contrainer(
                    label_set.get_start_states(),
                    label_set.get_end_states(),
                    label_set.get_transitions(),
                ),
            ),
        )
        tagger.to(device)

        collator = Collator(self.__tokenizer, label_set)

        train_dataloader = DataLoader(
            train_dataset,  # type:ignore
            collate_fn=collator,
            batch_size=self.__batch_size,
        )
        validation_dataloader = DataLoader(
            validation_dataset,  # type:ignore
            collate_fn=collator,
            batch_size=self.__batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            tagger.parameters(), lr=self.__learning_rate, weight_decay=0.0
        )
        schedular = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            SlantedTriangular(len(train_dataloader) * self.__num_epochs),
        )
        best_f1_score = float("-inf")
        best_tagger_state = io.BytesIO()

        for epoch in range(1, self.__num_epochs + 1):
            epoch_loss = 0.0
            tagger.train()

            for text_batch, tags_batch in train_dataloader:
                text_batch = cast(TextBatch, text_batch)
                tags_batch = cast(TagsBatch, tags_batch)

                text_batch.to(device)
                tags_batch.to(device)

                optimizer.zero_grad()

                mask = text_batch.mask

                log_potentials, _ = tagger(text_batch.tagger_inputs, mask)

                loss = expected_entity_ratio_loss(
                    log_potentials,
                    tags_batch.get_tag_bitmap(),
                    mask,
                    label_set.get_outside_index(),
                    self.__target_entity_ratio,
                    self.__entity_ratio_margin,
                    self.__balancing_coefficient,
                )
                loss.backward()

                torch.nn.utils.clip_grad_value_(
                    tagger.parameters(), clip_value=self.__gradient_clip_value
                )

                optimizer.step()
                schedular.step()

                epoch_loss += loss.item() * text_batch.size

            tagger.eval()
            metric = Metric()
            for text_batch, tags_batch in validation_dataloader:
                text_batch = cast(TextBatch, text_batch)
                tags_batch = cast(TagsBatch, tags_batch)

                text_batch.to(device)
                tags_batch.to(device)

                tag_indices = tagger.predict(text_batch.tagger_inputs, text_batch.mask)

                predictions = text_batch.create_char_based_tags(
                    tag_indices, label_set, tagger.padding_index
                )

                metric(predictions, tags_batch.char_based)

            scores = metric.get_scores()

            if best_f1_score < scores["f1_score"]:
                best_f1_score = scores["f1_score"]
                best_tagger_state.truncate(0)
                best_tagger_state.seek(0)
                torch.save(tagger.state_dict(), best_tagger_state)

            logger.info(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    **{f"validation_{key}": value for key, value in scores.items()},
                }
            )

        best_tagger_state.seek(0)
        tagger.load_state_dict(torch.load(best_tagger_state))

        return Recognizer(tagger, self.__tokenizer, label_set)
