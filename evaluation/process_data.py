import json
import os
import glob
import random
import re
import argparse

import numpy as np
import tqdm

import yaml
import sys


LANGS = ['cpp', 'java', 'js', 'python']

def extract(lang, config):
    """extract data from multipl-E (generation) and humaneval-X (reference)"""

    def extract_code(program, intent, tests):
        """extract the generation, remove the assertions"""
        program = program.replace(intent, '').replace(tests, '').strip()
        return program

    # read thu data for reference snippets
    with open(f'data/humaneval_x/humaneval_{lang}.jsonl', 'r') as f:
        idx_to_snippet = {}
        idx_to_full_prompt = {}
        for line in f:
            item = json.loads(line)
            idx = item['task_id'].split("/")[-1]
            idx_to_snippet[idx] = item['canonical_solution']
            idx_to_full_prompt[idx] = item['prompt']

    with open(f'data/humaneval_nl.json', 'r') as f:
        idx2nl = json.load(f)

    grade_results = []
    src_folder = 'data/multipl_e'
    for result_file in glob.glob(f'{src_folder}/{lang}-{config}/*.results.yaml'):
        correct = []
        print(result_file)
        with open(result_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        with open(result_file.replace('.results.yaml', '.yaml'), 'r') as f:
            data_source = yaml.load(f, Loader=yaml.FullLoader)
        intent = data_source['prompt']
        tests = data_source['tests']
        cur_grade = {'intent': intent}
        p_set = set()
        for idx, result in enumerate(data['results']):
            program = result['program']
            if program in p_set:
                continue
            p_set.add(program)
            program = extract_code(program, intent, tests)
            success = int(result['status'] == 'OK')
            cur_grade[f'{idx}'] = program
            cur_grade[f'grade-{idx}'] = {'execution': success}
            if success:
                correct.append(idx)
        task_idx = re.findall(r'HumanEval_(\d+)_', result_file)[0]
        assert task_idx.isdigit()
        cur_grade['snippet'] = [idx_to_snippet[task_idx]]
        cur_grade['prompt'] = idx_to_full_prompt[task_idx]
        cur_grade['simplified_intent'] = idx2nl[task_idx]
        cur_grade['task_id'] = task_idx
        grade_results.append([int(task_idx), cur_grade])

    grade_results = [x[1] for x in grade_results]
    os.makedirs(f'data/humaneval_{lang}_{config}', exist_ok=True)
    with open(f'data/humaneval_{lang}_{config}/human_grade.json', 'w') as f:
        json.dump(grade_results, f, indent=2)
    return grade_results

def clean(s):
    return s.replace("\n", " ")

def to_txt(lang, config):
    """convert to files of source, reference, target, one example each line"""

    folder = f'data/humaneval_{lang}_{config}'
    with open(f'{folder}/human_grade.json', 'r') as f:
        data = json.load(f)
        hyps = []
        refs = []
        srcs = []
        full_srcs = []
        meta_data = {}
        line_info = {}
        line_num = 0
        print(f"Number of examples: {len(data)}")

        for idx, item in enumerate(data):
            meta_data[idx] = {}
            for model in range(1, 100):
                model = str(model)
                if f'{model}' not in item:
                    continue

                meta_data[idx][model] = item[f'grade-{model}']
                for ref in item['snippet']:  # multiple refs per example
                    model_pred = item[model]
                    if model_pred.strip() == "":
                        model_pred = "placeholder"
                    hyps.append(model_pred)
                    refs.append(ref)
                    srcs.append(item['simplified_intent'])
                    full_srcs.append(item['prompt'])
                    line_info[line_num] = (idx, model)
                    line_num += 1

    print('Number of pairs: {}'.format(len(hyps)))

    with open(f'{folder}/humaneval_hyps.txt', 'w') as f:
        for hyp in hyps:
            assert hyp
            f.write(clean(hyp) + '\n')

    with open(f'{folder}/humaneval_refs.txt', 'w') as f:
        for ref in refs:
            assert ref
            f.write(clean(ref) + '\n')

    with open(f'{folder}/humaneval_srcs.txt', 'w') as f:
        for src in srcs:
            assert src
            f.write(clean(src) + '\n')

    with open(f'{folder}/humaneval_full_prompts.txt', 'w') as f:
        for src in full_srcs:
            assert src
            f.write(clean(src) + '\n')

    with open(f'{folder}/humaneval_meta.json', 'w') as f:
        meta_data['line_info'] = line_info
        json.dump(meta_data, f, indent=2)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='cpp')
    parser.add_argument(
        '--config', 
        help='the experimental config in multipl-E to generate the programs',
        type=str,
        default='davinci-0.8-keep')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = config()
    extract(args.lang, args.config)
    to_txt(args.lang, args.config)




