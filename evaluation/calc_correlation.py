"""
from
https://github.com/devjeetr/Re-assessing-automatic-evaluation-metrics-for-source-code-summarization-tasks/blob/main/scripts/kendalls_tau.py
"""
import argparse
import itertools
import json
import os.path

from scipy import stats
import operator
from collections import Counter
from typing import Iterable, Tuple, Union, Any
import numpy as np
import pandas as pd
from constants import HUMANEVAL_CV_SUBSETS, CONALA_CV_SUBSETS, MODELS_HUMANEVAL, MODELS_CONALA

CV_NUM = 5


def compute_human_ranks(scores: Tuple[float, float], threshold=25) -> Tuple[int, int]:
    """Given a pair of human direct assessment(DA) scores,
       computes the relative ranking. If the difference between
       the two scores is less than the provided threshold,
       the rank is the same.


    Args:
        scores (Tuple[int, int]): A tuple containing the 2 DA scores.
        threshold (int, optional): The threshold of the difference between two scores at which the
                                   the difference is considered (significant). Defaults to 25.

    Returns:
        Tuple[int, int]: The relative ranking of the provided scores
    """
    assert len(scores) == 2
    a, b = scores

    if (a == b) or abs(a - b) < threshold:
        return [1, 1]

    if a > b:
        return [1, 2]

    return [2, 1]


def get_index_pairs(df):
    pairs = list(itertools.combinations(df.index, 2))
    return [pair for pair in pairs if pair[0] != pair[1]]


# we use variant B from Stanchev et al. (Soft Penalization) for
# computing concordance/discordance/ties
comparison_variant_b = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "t"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_c = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "d"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "d"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_d = [
    (operator.lt, operator.lt, "c"),  # <, <
    (operator.lt, operator.eq, "t"),  # <, =
    (operator.lt, operator.gt, "d"),  # <, >
    (operator.eq, operator.lt, "t"),  # =, <
    (operator.eq, operator.eq, "c"),
    (operator.eq, operator.gt, "t"),  # =, >
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]


def compute_rank_pair_type(
        human_ranking: Iterable[Union[int, float]],
        metric_ranking: Iterable[Union[int, float]],
) -> Union[Any, Any, None]:
    comparison_table = comparison_variant_b

    for h_op, m_op, outcome in comparison_table:
        if h_op(*human_ranking) and m_op(*metric_ranking):
            return outcome

    return "-"


def kendalls_tau_darr(
        human_scores: np.ndarray,
        metric_scores: np.ndarray,
        threshold=0.001,
):
    """Computes the Kendall's Tau formulation for da-RR, as presented
    Stanchev et al., "Towards a Better Evaluation of Metrics for Machine Translation."

    It is given by:
                |Concordant - Discordant|
         \tau = -------------------------------
                |Concordant + Discordant + Ties|


     where:
         ╔═══════╦═════════╦═════════╦═════════╦═════════╗
         ║       ║         ║ metric  ║         ║         ║
         ╠═══════╬═════════╬═════════╬═════════╬═════════╣
         ║       ║         ║ s1 < s2 ║ s1 = s2 ║ s1 > s2 ║
         ╠═══════╬═════════╬═════════╬═════════╬═════════╣
         ║ human ║ s1 < s2 ║ Conc    ║ Tie     ║ Disc    ║
         ║       ╠═════════╬═════════╬═════════╬═════════╣
         ║       ║ s1 = s2 ║ -       ║ -       ║ -       ║
         ║       ╠═════════╬═════════╬═════════╬═════════╣
         ║       ║ s1 > s2 ║ Disc    ║ Tie     ║ Conc    ║
         ╚═══════╩═════════╩═════════╩═════════╩═════════╝

     args:
         human_rankings - 2d numpy array of human rankings
         metric_rankings - 2d numpy array of metric rankings, corresponding
                           to the human rankings
        threshold - difference in human score that above which human scores are treated
                    to be different. For example, with threshold of 25, and s1=50 and s2=60,
                    at a difference of 10 < 25, the two scores are treated as equal.
    """
    counts = Counter()
    for h_pair, m_pair in zip(human_scores, metric_scores):
        h_rank = compute_human_ranks(h_pair, threshold)
        m_rank = compute_human_ranks(m_pair, threshold)
        pair_type = compute_rank_pair_type(h_rank, m_rank)
        counts[pair_type] += 1
    concordant_pairs = counts["c"]
    discordant_pairs = counts["d"]
    ties = counts["t"]
    tau = (concordant_pairs - discordant_pairs) / (
            concordant_pairs + discordant_pairs + ties
    )
    return {
        "tau": tau,
        "concordant": concordant_pairs,
        "discordant": discordant_pairs,
        "ties": ties,
    }


def get_human_metric_paired_scores(file_name, meta_file, metric_key='f1',
                                   models=None,
                                   overwrite=False):
    with open(file_name, 'r') as f:
        metric_scores = json.load(f)
        if 'fmeteor' in metric_scores:
            metric_scores['fm'] = metric_scores.pop('fmeteor')
        metric_scores = metric_scores[metric_key]

    with open(meta_file, 'r') as f:
        meta = json.load(f)
        line_info = meta['line_info']

    for line_idx, score in enumerate(metric_scores):
        line_idx = str(line_idx)
        sample_idx, model_name = line_info[line_idx]
        meta[str(sample_idx)][f"pred.{model_name}"] = score

    # enumerate pairs
    score_file = file_name.replace('.score.json', f'.{metric_key}.pairscore.tau.npz')
    if not overwrite and os.path.exists(score_file):
        pairs = np.load(score_file)
        human_scores = pairs['human_scores']
        metric_scores = pairs['metric_scores']
        print(f"Loaded {score_file}")
    else:
        human_scores = []
        metric_scores = []
        for sample_idx, scores in meta.items():
            if sample_idx == 'line_info':
                continue
            sample_human_scores = []
            sample_metric_scores = []
            for m1 in models:
                for m2 in models:
                    if m1 == m2:
                        continue
                    if m1 not in scores or m2 not in scores:
                        continue
                    hs1 = np.mean(list(scores[m1].values()))
                    hs2 = np.mean(list(scores[m2].values()))
                    ms1 = scores[f"pred.{m1}"]
                    ms2 = scores[f"pred.{m2}"]
                    sample_human_scores.append([hs1, hs2])
                    sample_metric_scores.append([ms1, ms2])
            if sample_human_scores:
                human_scores.append(sample_human_scores)
                metric_scores.append(sample_metric_scores)
        human_scores = np.array(human_scores, dtype=object)
        metric_scores = np.array(metric_scores, dtype=object)
        # np.savez(score_file, human_scores=human_scores, metric_scores=metric_scores)
        # print(f"Saved {score_file}")

    print(human_scores.shape, metric_scores.shape)
    return human_scores, metric_scores


def get_human_metric_scores(file_name, 
                            meta_file, 
                            metric_key='f1',
                            models=None):
    with open(file_name, 'r') as f:
        metric_scores = json.load(f)
        if 'fmeteor' in metric_scores:
            metric_scores['fm'] = metric_scores.pop('fmeteor')
        metric_scores = metric_scores[metric_key]


    with open(meta_file, 'r') as f:
        meta = json.load(f)
        line_info = meta['line_info']

    for line_idx, score in enumerate(metric_scores):
        line_idx = str(line_idx)
        sample_idx, model_name = line_info[line_idx]
        meta[str(sample_idx)][f"pred.{model_name}"] = score

    # enumerate pairs
    human_scores = []
    metric_scores = []
    for sample_idx, scores in meta.items():
        if sample_idx == 'line_info':
            continue
        sample_human_scores = []
        sample_metric_scores = []
        for m in models:
            if m not in scores:
                continue
            hs = np.mean(list(scores[m].values()))
            ms = scores[f"pred.{m}"]
            sample_human_scores.append(hs)
            sample_metric_scores.append(ms)
        if sample_human_scores:
            human_scores.append(sample_human_scores)
            metric_scores.append(sample_metric_scores)
    human_scores = np.array(human_scores, dtype=object)
    metric_scores = np.array(metric_scores, dtype=object)
    print(human_scores.shape, metric_scores.shape)
    return human_scores, metric_scores


def best_config(file_name='',
                scores=None,
                blacklist=None,
                ds='humaneval'):

    if scores is None:
        with open(file_name, 'r') as f:
            scores = json.load(f)

    if blacklist:  # remove blacklisted configs
        for k in list(scores):
            if any([b in k for b in blacklist]):
                del scores[k]

    # sort scores by name
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[0])}

    dev_tags = []
    dev_taus = []
    dev_pps = []
    dev_sps = []
    for k, v in scores.items():
        if 'dev' in k:
            dev_tags.append(k)
            dev_taus.append(np.mean(v[0]))
            dev_pps.append(np.mean(v[1]))
            dev_sps.append(np.mean(v[2]))

    for i in range(len(dev_tags)):
        tag = dev_tags[i]
        print(f'{tag}\t'
              f'{(np.mean(scores[tag][0]) + np.mean(scores[tag][2])) / 2:.4f}\t')
        
    for i in range(len(dev_tags)):
        tag = dev_tags[i].replace('dev', 'test')
        print(f'{tag}\t'
              f'{np.mean(scores[tag][0]):.4f}({np.std(scores[tag][0]):.4f})\t'
              f'{np.mean(scores[tag][1]):.4f}({np.std(scores[tag][1]):.4f})\t',
              f'{np.mean(scores[tag][2]):.4f}({np.std(scores[tag][2]):.4f})')

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_folder', type=str, default='')
    parser.add_argument('--d_prefix', type=str, default='')
    parser.add_argument('--result_file', type=str, default='')
    parser.add_argument('--metric_key', type=str, default='fm')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = config()

    d_folder = args.d_folder
    d_prefix = args.d_prefix
    result_file = f'{d_folder}/{args.result_file}'
    metric_key = args.metric_key
    
    if 'humaneval' in d_folder:
        models = MODELS_HUMANEVAL
        cv_subsets = HUMANEVAL_CV_SUBSETS
        ds = 'humaneval'
    elif 'conala' in d_folder:
        models = MODELS_CONALA
        cv_subsets = CONALA_CV_SUBSETS
        ds = 'conala'
    else:
        raise ValueError('Unknown dataset')
    

    meta_file = f'{d_folder}/{d_prefix}_meta.json'
    save_file = f"{d_folder}/{d_prefix}_results.json"
    results = {}


    human_paired_scores, \
    metric_paired_scores = get_human_metric_paired_scores(
        result_file,
        meta_file,
        metric_key=metric_key,
        models=models,
        overwrite=True)
    
    human_scores, \
    metric_scores = get_human_metric_scores(
        result_file,
        meta_file,
        metric_key=metric_key,
        models=models)

    assert human_paired_scores.shape[0] == human_scores.shape[0]
    N = human_scores.shape[0]
    dev_avg_tau = []
    dev_avg_pp = []
    dev_avg_sp = []
    test_avg_tau = []
    test_avg_pp = []
    test_avg_sp = []

    # 5 cross-validation
    for cv_index in range(CV_NUM):
        subset = cv_subsets[cv_index]

        dev_h, dev_m = human_paired_scores[subset], metric_paired_scores[subset]
        # cat on the first dimension
        dev_h = np.concatenate(dev_h, axis=0)
        dev_m = np.concatenate(dev_m, axis=0)

        dev_tau = kendalls_tau_darr(dev_h, dev_m)
        test_h, test_m = human_paired_scores[~subset], metric_paired_scores[~subset]
        test_h = np.concatenate(test_h, axis=0)
        test_m = np.concatenate(test_m, axis=0)
        test_tau = kendalls_tau_darr(test_h, test_m)
        dev_avg_tau.append(dev_tau['tau'])
        test_avg_tau.append(test_tau['tau'])

        # pearson
        dev_h, dev_m = human_scores[subset], metric_scores[subset]
        dev_h = np.concatenate(dev_h, axis=0)
        dev_m = np.concatenate(dev_m, axis=0)
        dev_pp = stats.pearsonr(dev_h, dev_m)
        dev_sp = stats.spearmanr(dev_h, dev_m)
        test_h, test_m = human_scores[~subset], metric_scores[~subset]
        test_h = np.concatenate(test_h, axis=0)
        test_m = np.concatenate(test_m, axis=0)
        test_pp = stats.pearsonr(test_h, test_m)
        test_sp = stats.spearmanr(test_h, test_m)
        dev_avg_pp.append(dev_pp[0])
        dev_avg_sp.append(dev_sp.correlation)
        test_avg_pp.append(test_pp[0])
        test_avg_sp.append(test_sp.correlation)

    run_tag = result_file.replace(f"{d_prefix}_", "").replace(".score.json", "") + f".{metric_key}"
    results[f'{run_tag}_dev'] = [list(dev_avg_tau),
                                    list(dev_avg_pp),
                                    list(dev_avg_sp)]
    results[f'{run_tag}_test'] = [list(test_avg_tau),
                                    list(test_avg_pp),
                                    list(test_avg_sp)]

    with open(save_file, 'w+') as f:
        json.dump(results, f, indent=2)

    best_config(file_name=save_file, ds=ds)



