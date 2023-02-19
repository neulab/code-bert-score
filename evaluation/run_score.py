import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_bert_score import score, utils
import json
import re
import pickle
import argparse


def clean_code(code):
    code = code.replace('\'', '"')
    tokens = [t for t in code.split(' ') if t]
    return ' '.join(tokens)

def main(
            model_name, 
            device, 
            d_folder, 
            d_prefix,
            lang,
            num_layers=None, 
            no_punc=True, 
            use_source=True,
            batch_size=64, 
            verbose=False,
            overwrite=False, 
            idf_path='',
            source_file_suffix='srcs.txt'):

    with open(f"{d_folder}/{d_prefix}_hyps.txt") as f:
        cands = [clean_code(line.strip()) for line in f]
        cands = [x if x.strip() else '[wrong]' for x in cands]
        print(f'empty candidate: {cands.count("[wrong]")}')

    with open(f"{d_folder}/{d_prefix}_refs.txt") as f:
        refs = [clean_code(line.strip()) for line in f]
        refs = [x if x.strip() else '[wrong]' for x in refs]
        print(f'empty reference: {refs.count("[wrong]")}')

    if use_source:
        with open(f'{d_folder}/{d_prefix}_{source_file_suffix}') as f:
            srcs = []
            for line in f:
                line = line.strip().replace("\n", " ")
                srcs.append(f"# {line}")
        assert all(srcs)
    else:
        srcs = None

    if idf_path != '':
        with open(idf_path, 'rb') as f:
            idf_dict = pickle.load(f)
    else:
        idf_dict = None

    hashname = utils.get_hash(model=model_name,
                              num_layers=num_layers,
                              idf=idf_dict is not None,
                              rescale_with_baseline=False,
                              use_custom_baseline=False).split("/")[-1]
    
    save_file = f"{d_folder}/{d_prefix}_{hashname}.score.json"

    if os.path.exists(save_file) and not overwrite:
        print("Results already exists")
        return

    assert srcs is None or len(cands) == len(srcs)
    assert len(cands) == len(refs)
    print("Number of samples:", len(cands))

    (P, R, F, FM), _ = score(cands, refs,
                                lang=lang,
                                device=device,
                                idf = idf_dict,
                                model_type=model_name,
                                return_hash=True,
                                no_punc=no_punc,
                                num_layers=num_layers,
                                sources=srcs,
                                verbose=verbose,
                                batch_size=batch_size)

    print(f"{hashname}: P={P.mean().item():.6f}"
          f"R={R.mean().item():.6f}"
          f"F={F.mean().item():.6f}"
          f"FM={FM.mean().item():.6f}")

    with open(save_file, "w+") as f:
        d = {'precision': P.tolist(), 'recall': R.tolist(), 'f1': F.tolist(), 'fm': FM.tolist()}
        json.dump(d, f, indent=2)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--d_folder", type=str)
    parser.add_argument("--d_prefix", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--source_file_suffix", default='srcs.txt')
    parser.add_argument("--idf_path", type=str)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))
    return args

if __name__ == '__main__':
    args = config()
    main(
            model_name=args.model,
            device=args.device,
            d_folder = args.d_folder,
            d_prefix = args.d_prefix,
            no_punc=True,
            num_layers=args.layer,
            batch_size=64,
            use_source=True,
            lang=args.lang,
            idf_path=args.idf_path,
            overwrite=False,
            source_file_suffix=args.source_file_suffix
        )
