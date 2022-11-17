from argparse import ArgumentParser
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from functools import partial
from collections import defaultdict
from math import log

from code_bert_score.utils import get_idf_dict

def default_count(num_examples):
    return log((num_examples + 1) / (1))

if __name__ == '__main__':
    parser = ArgumentParser()
    # The language is the only required argument:
    parser.add_argument('--subset', required=True, help="The language: java, js, python, cpp, go")
    # Default valued not required arguments:
    parser.add_argument('--dataset', required=False, default="THUDM/humaneval-x")
    parser.add_argument('--field', required=False, default="canonical_solution")
    parser.add_argument('--split', required=False, default='test')
    parser.add_argument('--tokenizer', required=False, default='microsoft/codebert-base-mlm')
    parser.add_argument('--nthreads', required=False, type=int, default=4)
    parser.add_argument('--output', required=False, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f'{args.subset}_{args.split}_idf.pkl'
    
    print(f'Loading dataset {args.dataset}...')
    dataset = datasets.load_dataset(path=args.dataset, name=args.subset, split=args.split)
    print(f'Loading tokenizer {args.tokenizer}...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f'Creating IDF dict for {args.subset}...')
    idf_dict = get_idf_dict(arr=dataset[args.field], tokenizer=tokenizer, nthreads=args.nthreads)
    
    idf_dict = dict(idf_dict)
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    default_value = default_count(len(dataset))
    for i in range(tokenizer.vocab_size):
        if i not in idf_dict:
            idf_dict[i] = default_value

    with open(args.output, 'wb') as f:
        pickle.dump(idf_dict, f)
    print(f'Done! Saved to {args.output}')
