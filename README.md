# pytorch-partial-tagger

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17GcpKmYn49bMM-mCZhiuRu8dkPZ0WXuD?usp=sharing)

`pytorch-partial-tagger` is a Python library for building a sequence tagger, specifically for the common NLP task Named Entity Recognition, with a partially annotated dataset in PyTorch.
You can build your own tagger using a distantly-supervised dataset obtained from unlabled text data and a dictionary that maps surface names to their entity type.
The algorithm of this library is based on Effland and Collins. (2021).


## Usage

Import all dependencies first:

```py
import torch
from sequence_label import SequenceLabel

from partial_tagger.metric import Metric
from partial_tagger.utils import create_trainer
```

Prepare your own datasets. Each item of dataset must have a pair of a string and a sequence label.
A string represents text that you want to assign a label, which is defined as `text` below.
A sequence label represent a set of a character-based tag, which has a start, a length, and a label, which are defined as `label` below.
A start represents a position in the text where a tag starts.
A length represents a distance in the text between the beginning of a tag and the end of a tag.
A label represents what you want to assign to a span of the text defined by a start and a length.

```py
text = "Tokyo is the capital of Japan."
label = SequenceLabel.from_dict(
    tags=[
        {"start": 0,  "end": 5, "label": "LOC"},  # Tag for Tokyo
        {"start": 24,  "end": 29, "label": "LOC"},  # Tag for Japan
    ],
    size=len(text),
)

train_dataset = [(text, label), ...]
validation_dataset = [...]
test_dataset = [...]
```

Here, you will train your tagger and evaluate its performance.
You will train it through an instance of `Trainer`, which you get by calling `create_trainer`.
After a training, you will get an instance of `Recognizer` which predicts character-based tags from given texts.
You will evaluate the performance of your tagger using an instance of `Metric` as follows.


```py

device = torch.device("cuda")

trainer = create_trainer()
recognizer = trainer(train_dataset, validation_dataset, device)

texts, ground_truths = zip(*test_dataset)

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

## Documentation

For details about the `pytorch-partial-tagger` API,  see the [documentation](https://pytorch-partial-tagger.readthedocs.io/en/latest/).

## References

- Thomas Effland and Michael Collins. 2021. [Partially Supervised Named Entity Recognition via the Expected Entity Ratio Loss](https://aclanthology.org/2021.tacl-1.78/). _Transactions of the Association for Computational Linguistics_, 9:1320–1335.
