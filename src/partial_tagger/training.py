from __future__ import annotations

import io
import logging
from logging import Logger
from typing import cast

import torch
from torch.utils.data import DataLoader

from .crf import functional as F
from .data import LabelSet
from .data.batch import Batch, CharBasedTagsCollection, Dataset
from .recognizer import Recognizer
from .utils import Metric, create_collator, create_tagger


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
        model_name: str = "roberta-base",
        batch_size: int = 15,
        num_epochs: int = 20,
        learning_rate: float = 2e-5,
        gradient_clip_value: float = 5.0,
        padding_index: int = -1,
        unknown_index: int = -100,
        tokenizer_args: dict | None = None,
    ):
        self.__model_name = model_name
        self.__batch_size = batch_size
        self.__num_epochs = num_epochs
        self.__learning_rate = learning_rate
        self.__gradient_clip_value = gradient_clip_value
        self.__padding_index = padding_index
        self.__unknown_index = unknown_index
        self.__tokenizer_args = tokenizer_args

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

        # Create a tagger
        tagger = create_tagger(self.__model_name, label_set, self.__padding_index)
        tagger.to(device)

        # Create a collator
        collator = create_collator(self.__model_name, label_set, self.__tokenizer_args)

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
        best_f1_score = 0.0
        best_tagger_state = io.BytesIO()

        for epoch in range(1, self.__num_epochs + 1):
            epoch_loss = 0.0
            tagger.train()

            for batch, ground_truths in train_dataloader:
                batch = cast(Batch, batch)
                ground_truths = cast(CharBasedTagsCollection, ground_truths)

                optimizer.zero_grad()

                log_potentials, _ = tagger(
                    batch.get_tagger_inputs(device), batch.get_mask(device)
                )

                tags_bitmap = batch.create_tag_bitmap(
                    ground_truths, device, self.__padding_index, self.__unknown_index
                )

                loss = expected_entity_ratio_loss(
                    log_potentials,
                    tags_bitmap,
                    batch.get_mask(device),
                    label_set.get_outside_index(),
                )
                loss.backward()

                torch.nn.utils.clip_grad_value_(
                    tagger.parameters(), clip_value=self.__gradient_clip_value
                )

                optimizer.step()
                schedular.step()

                epoch_loss += loss.item() * batch.size

            tagger.eval()
            metric = Metric()
            for batch, ground_truths in validation_dataloader:
                batch = cast(Batch, batch)
                ground_truths = cast(CharBasedTagsCollection, ground_truths)

                tag_indices = tagger.predict(
                    batch.get_tagger_inputs(device), batch.get_mask(device)
                )

                predictions = batch.create_char_based_tags(
                    tag_indices, self.__padding_index
                )

                metric(predictions, ground_truths)

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

        return Recognizer(tagger, collator.batch_factory, self.__padding_index)
