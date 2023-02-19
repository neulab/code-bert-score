# Evaluation of CodeBertScore
This folder contains the full evaluation pipeline on the correlation with functional correctness 

```bash
cd evaluation
LANG=java # cpp, python, js
MODEL_LANG=java # if LANG is js, use javascript
LAYER=7
```
## 1. Data preparation
We construct multilingual humaneval dataset from [multipl-E](https://github.com/nuprl/MultiPL-E) and [humaneval-x](https://github.com/THUDM/CodeGeeX/tree/main/codegeex/benchmark/humaneval-x)
```bash
python process_data.py \
    --lang LANG \
    --config davinci-0.8-keep 
```
This script will take
1. generation results provided by multipl-e ([example](./data/multipl_e/java-davinci-0.8-keep/))
2. reference code in the corresponding language from humaneval-x ([example](./data/humaneval_x/humaneval_java.jsonl))
and construct text file of source, reference and target ([example](./data/humaneval_java_davinci-0.8-keep/humaneval_refs.txt))

## 2. Calculate CodeBertScore
```bash
python run_score.py \
    --lang $LANG \
    --model neulab/codebert-$MODEL_LANG \
    --device cuda:0 \
    --d_folder data/humaneval_$LANG_davinci-0.8-keep \
    --d_prefix humaneval \
    --idf_path data/idf/$LANG_idf.pkl \
    --layer $LAYER 
```
The detailed configurations for each language are provided [here](run_score.sh)

## 3. Calculate correlation with functional correctness
```bash
python calculate_correlation.py \
    --lang $LANG \
    --d_folder data/humaneval_$LANG_davinci-0.8-keep \
    --d_prefix humaneval \
    --result_file humaneval_codebert-$MODEL_LANG_L$LAYER_idf.score.json
```
It will output the kental-tau, spearman and pearson correlation with functional correctness.
