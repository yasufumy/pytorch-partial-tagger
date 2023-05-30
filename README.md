# pytorch-partial-tagger

 This is a library to build a CRF tagger for a partially annotated dataset in PyTorch. You can build your own NER tagger only from dictionary. The algorithm of this tagger is based on Effland and Collins. (2021).


## Usage

Import all dependencies first:

```py
import torch

from partial_tagger.data import CharBasedTags
from partial_tagger.training import Trainer
from partial_tagger.utils import Metric, create_tag
```

Prepare your own datasets.
Each item of dataset must have a string and tags. A string represents `text` below.
Tags represent a collection of tags, where each tag has a start, a length, and a label, which are defined as `tags` below.
A start represents a position in `text` where a tag starts.
A length represents a distance in `text` between the beginning of a tag and the end of a tag.
A label represents what you want to assign to a span of `text` defined by a start and a length.

```py
from partial_tagger.utils import create_tags, CharBasedTags


text = "Tokyo is the capital of Japan."
tags = CharBasedTags(
    (
        create_tag(start=0, length=5, label="LOC"),  # Tag for Tokyo
        create_tag(start=24, length=5, label="LOC")  # Tag for Japan
    ),
    text
)
train_dataset = [(text, tags), ...]
validation_dataset = [...]
test_dataset = [...]
```

Here, you would train your tagger and evaluate its performance.

You could train your own tagger by initializing `Trainer` and passing datasets to it.
After training, `trainer` gives you `Recognizer` object which predicts character-based tags from given texts.

You could evaluate the performance of your tagger using `Metric` as below.


```py

device = torch.device("cuda")

trainer = Trainer()
recognizer = trainer(train_dataset, validation_dataset, device)

texts, ground_truths = zip(*test_dataset, strict=True)

batch_size = 15
predictions = recognizer(texts, batch_size, device)

metric = Metric()
metric(predictions, ground_truths)

print(metric.get_scores())  # Display F1-score, Precision, Recall
```

## Installation

```bash
pip install pytorch-partial-tagger
```

## References

- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). _Transactions of the Association for Computational Linguistics_, 9:1320–1335.
