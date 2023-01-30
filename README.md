# CodeBERTScore
<!-- [![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![PyPI version bert-score](https://badge.fury.io/py/bert-score.svg)](https://pypi.python.org/pypi/bert-score/) [![Downloads](https://pepy.tech/badge/bert-score)](https://pepy.tech/project/bert-score) [![Downloads](https://pepy.tech/badge/bert-score/month)](https://pepy.tech/project/bert-score/month) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  -->

An Automatic Evaluation Metric for Code, based on [BERTScore](https://arxiv.org/abs/1904.09675).

This repository is based on the code of [BERTScore](https://github.com/Tiiiger/bert_score), and we are grateful to the authors for releasing their code.

### Background: BERTScore
BERTScore leverages the pre-trained contextual embeddings from BERT and matches
words in candidate and reference sentences by cosine similarity.
It has been shown to correlate with human judgment on sentence-level and
system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be
useful for evaluating different language generation tasks.

For an illustration, BERTScore recall can be computed as
![](./bert_score.png "BERTScore")

## Usage
```
import code_bert_score
pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='python')
```
Where `pred_results` is a 4-tuple of `(precision, recall, F1, F3)`, where each is a 1-D tensor of scores for each prediction-reference pair. `F3` is similar to the well-known `F1` score, that considers recall 3 times as important as precision. See the [definition on Wikipedia](https://en.wikipedia.org/wiki/F-score#F%CE%B2_score).

## Additional Features

* We found that sometimes more accurate results are achieved using the `no_punc=True` argument, that encodes the *entire* inputs, but measures the similarity only non-punctuation and non-whitespace tokens:

```
pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='python', no_punc=True)
```

* We found that in NL->Code problems, more accurate results are achieved by encoding the NL source with the code prediction, but then measuring similarity only for the encoded code:

```
pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='python', sources=sources)
```

* We also found that using Inverse Document Frequencies improve the results, similarly to the original BERTScore. We included an example script that shows how to precompute them here [compute_idf.py](compute_idf.py). Then, the resulting dictionary can be used with the argument `idf=idf_dict`.

* Tuning the layer that the similarity is computed from is also helpful, using `num_layers=N` where `N` is between 5-9.

See also our [example.py](./example.py) and the original BERTScore [demo notebook](./example/Demo.ipynb).

## Backend Model
We fine-tuned the `microsoft/codebert-base-mlm` model for 1,000,000 steps (with `batch_size=32`) on several languages separately.

We released the following models to the Huggingface hub:
* `anonymized/codebert-python` (the default model for `lang='python'`)
* `anonymized/codebert-javascript` (the default model for `lang='javascript'` or `'js'`)
* `anonymized/codebert-c` (the default model for `lang='c'`)
* `anonymized/codebert-cpp` (the default model for `lang='cpp'` or `'c++'`)
* `anonymized/codebert-java` (the default model for `lang='java'`)

all other languages currently use the `microsoft/codebert-base-mlm` model.

## Training
The [`run_mlm.py`](./run_mlm.py) script can be used to fine-tune the base model `microsoft/codebert-base-mlm` on specific languages.

## Human Evaluation

We are in the process of performing a human evaluation that measures the correlation between CodeBERTScore and human judgement.

