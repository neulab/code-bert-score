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
Where `pred_results` is a 3-tuple of `(precision, recall, F1)`, where each is a 1-D tensor of scores for each prediction-reference pair.

We found that sometimes more accurate results are achieved using the `no_punc=True` argument, that encodes the *entire* inputs, but measures the similarity only non-punctuation and non-whitespace tokens:

```
pred_results = code_bert_score.score(cands=predictions, refs=refs, lang='python', no_punc=True)
```


See also our [example.py](./example.py) and the original BERTScore [demo notebook](./example/Demo.ipynb).

## Backend Model
Currently, all languages use the `microsoft/codebert-base-mlm` model.
We are in the process of releasing fine-tuned models for a variety of programming languages such as Python, Java, JavaScript, C, and C++.

## Training
The [`run_mlm.py`](./run_mlm.py) script can be used to fine-tune the base model `microsoft/codebert-base-mlm` on specific languages.

## Human Evaluation

We are in the process of performing a human evaluation that measures the correlation between CodeBERTScore and human judgement.

<!-- If you find this repo useful, please cite:
```
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
``` -->

<!-- ### Installation
* Python version >= 3.6
* PyTorch version >= 1.0.0

Install from pypi with pip by 

```sh
pip install bert-score
```
Install latest unstable version from the master branch on Github by:
```
pip install git+https://github.com/Tiiiger/code_bert_score
```

Install it from the source by:
```sh
git clone https://github.com/Tiiiger/code_bert_score
cd code_bert_score
pip install .
```
and you may test your installation by:
```
python -m unittest discover
```

### Usage -->


<!-- #### Python Function

On a high level, we provide a python function `code_bert_score.score` and a python object `code_bert_score.BERTScorer`.
The function provides all the supported features while the scorer object caches the BERT model to faciliate multiple evaluations.
Check our [demo](./example/Demo.ipynb) to see how to use these two interfaces. 
Please refer to [`code_bert_score/score.py`](./code_bert_score/score.py) for implementation details.

Running BERTScore can be computationally intensive (because it uses BERT :p).
Therefore, a GPU is usually necessary. If you don't have access to a GPU, you
can try our [demo on Google Colab](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q)

#### Command Line Interface (CLI)
We provide a command line interface (CLI) of BERTScore as well as a python module. 
For the CLI, you can use it as follows:
1. To evaluate English text files:

We provide example inputs under `./example`.

```sh
bert-score -r example/refs.txt -c example/hyps.txt --lang en
```
You will get the following output at the end:

roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0) P: 0.957378 R: 0.961325 F1: 0.959333

where "roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)" is the hash code.

Starting from version 0.3.0, we support rescaling the scores with baseline scores

```sh
bert-score -r example/refs.txt -c example/hyps.txt --lang en --rescale_with_baseline
```
You will get:

roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)-rescaled P: 0.747044 R: 0.770484 F1: 0.759045 

This makes the range of the scores larger and more human-readable. Please see this [post](./journal/rescale_baseline.md) for details.

When having multiple reference sentences, please use
```sh
bert-score -r example/refs.txt example/refs2.txt -c example/hyps.txt --lang en
```
where the `-r` argument supports an arbitrary number of reference files. Each reference file should have the same number of lines as your candidate/hypothesis file. The i-th line in each reference file corresponds to the i-th line in the candidate file.


2. To evaluate text files in other languages:

We currently support the 104 languages in multilingual BERT ([full list](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)).

Please specify the two-letter abbreviation of the language. For instance, using `--lang zh` for Chinese text. 

See more options by `bert-score -h`.


3. To load your own custom model:
Please specify the path to the model and the number of layers to use by `--model` and `--num_layers`.
```sh
bert-score -r example/refs.txt -c example/hyps.txt --model path_to_my_bert --num_layers 9
```


4. To visualize matching scores:
```sh
bert-score-show --lang en -r "There are two bananas on the table." -c "On the table are two apples." -f out.png
```
The figure will be saved to out.png.


#### Practical Tips

* Report the hash code (e.g., `roberta-large_L17_no-idf_version=0.3.0(hug_trans=2.3.0)-rescaled`) in your paper so that people know what setting you use. This is inspired by [sacreBLEU](https://github.com/mjpost/sacreBLEU). Changes in huggingface's transformers version may also affect the score (See [issue #46](https://github.com/Tiiiger/code_bert_score/issues/46)).
* Unlike BERT, RoBERTa uses GPT2-style tokenizer which creates addition " " tokens when there are multiple spaces appearing together. It is recommended to remove addition spaces by `sent = re.sub(r' +', ' ', sent)` or `sent = re.sub(r'\s+', ' ', sent)`.
* Using inverse document frequency (idf) on the reference
  sentences to weigh word importance  may correlate better with human judgment.
  However, when the set of reference sentences become too small, the idf score 
  would become inaccurate/invalid.
  We now make it optional. To use idf,
  please set `--idf` when using the CLI tool or
  `idf=True` when calling `code_bert_score.score` function.
* When you are low on GPU memory, consider setting `batch_size` when calling
  `code_bert_score.score` function.
* To use a particular model please set `-m MODEL_TYPE` when using the CLI tool
  or `model_type=MODEL_TYPE` when calling `code_bert_score.score` function. 
* We tune layer to use based on WMT16 metric evaluation dataset. You may use a
  different layer by setting `-l LAYER` or `num_layers=LAYER`. To tune the best layer for your custom model, please follow the instructions in [tune_layers](tune_layers) folder.
* __Limitation__: Because BERT, RoBERTa, and XLM with learned positional embeddings are pre-trained on sentences with max length 512, BERTScore is undefined between sentences longer than 510 (512 after adding \[CLS\] and \[SEP\] tokens). The sentences longer than this will be truncated. Please consider using XLNet which can support much longer inputs.

### Default Behavior

#### Default Model
| Language  | Model                        |
|:---------:|:----------------------------:|
| en        | roberta-large                |
| en-sci    | scibert-scivocab-uncased     |
| zh        | bert-base-chinese            |
| others    | bert-base-multilingual-cased |

#### Default Layers
Please see this [Google sheet](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit?usp=sharing) for the supported models and their performance.

### Acknowledgement
This repo wouldn't be possible without the awesome
[bert](https://github.com/google-research/bert), [fairseq](https://github.com/pytorch/fairseq), and [transformers](https://github.com/huggingface/transformers). -->
