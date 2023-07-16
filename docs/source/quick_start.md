# Quick Start

We will replicate the experiment in [Effland and Collins. '21](https://aclanthology.org/2021.tacl-1.78/).

## Download datasets

Here, we will download datasets provided on [teffland/ner-expected-entity-ratio](https://github.com/teffland/ner-expected-entity-ratio/tree/main) below. We use the datasets for the experimental setting Non-Native Speaker Scenario (NNS): Recall=50%, Precision=90% in [Effland and Collins. '21](https://aclanthology.org/2021.tacl-1.78/).

```bash
curl -LO https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.train_r0.5_p0.9.jsonl
curl -LO https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.dev.jsonl
curl -LO https://raw.githubusercontent.com/teffland/ner-expected-entity-ratio/main/data/conll2003/eng/entity.test.jsonl
```

## Import all dependencies 

```py
import json
import logging
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch

from partial_tagger.metric import Metric
from partial_tagger.utils import create_tag,  create_trainer
```

## Define utility functions

We will prepare two utility functions. One is for fixing random state and the other is for displaying training logs.


```py
def fix_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class JSONAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return json.dumps(msg), kwargs


@contextmanager
def get_logger(log_name, log_file):
    logger = logging.getLogger(log_name)
    logger.propagate = False

    logger.setLevel(logging.INFO)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]

    for handler in handlers:
        logger.addHandler(handler)

    try:
        yield JSONAdapter(logger, {})
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
```

## Prepare datasets

We will prepare our datasets. Each item of the datasets must have a string and tags. A string represents `text` below. Tags represent the collection of a tag, where each tag has a start, a length, and a label, which are defined as `tags` below. A start represents a position in text where a tag starts. A length represents a distance in text between the beginning of a tag and the end of a tag. A label represents what you want to assign to a span of text defined by a start and a length.


```py
def load_dataset(path: str):
    with open(path) as f:
        dataset = []

        for line in f:
            data = json.loads(line.rstrip())

            text = " ".join(data["tokens"])

            mapping = {}
            now = 0
            for i, token in enumerate(data["tokens"]):
                mapping[i] = now
                now += len(token) + 1  # Add one for a space

            tags = {
                create_tag(
                    mapping[annotation["start"]],
                    len(annotation["mention"]),
                    annotation["type"],
                )
                for annotation in data["gold_annotations"]
            }

            dataset.append((text, tags))

    return dataset


train_dataset = load_dataset("entity.train_r0.5_p0.9.jsonl")
dev_dataset = load_dataset("entity.dev.jsonl")
test_dataset = load_dataset("entity.test.jsonl")
```

## Train our tagger

We will train our tagger by initializing Trainer and passing datasets to it. After training, trainer gives you an instance of `Recognizer` which predicts character-based tags from given texts. 


```py
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "roberta-base"
dropout = 0.2
batch_size = 15
num_epochs = 20
learning_rate = 2e-5
gradient_clip_value = 5.0
padding_index = -1
unknown_index = -100
train_log_file = "log.jsonl"
tokenizer_args = {
    "padding": True,
    "return_tensors": "pt",
    "return_offsets_mapping": True
}

fix_state(seed)

trainer = create_trainer(
    model_name,
    dropout,
    batch_size,
    num_epochs,
    learning_rate,
    gradient_clip_value,
    padding_index,
    encoder_type="default",
)

with get_logger(f"{__name__}.train", train_log_file) as logger:
  recognizer = trainer(
      train_dataset,
      dev_dataset,
      device,
      logger
  )
```


## Evaluate our tagger

We will evaluate the performance of our tagger using Metric as below.

```py
texts, ground_truths = zip(*test_dataset)

predictions = recognizer(texts, batch_size, device)

metric = Metric()
metric(predictions, ground_truths)

test_scores_file = "scores.json"
with get_logger(f"{__name__}.evaluate", test_scores_file) as logger:
  logger.info({f"test_{key}": value for key, value in metric.get_scores().items()})
```
