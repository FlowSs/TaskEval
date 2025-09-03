import os
import argparse
import json

import pandas as pd

from ast_analyzer import count_code_elements
from radon.raw import analyze
import tqdm

def ast_parse_f(x, loc):
    ast_parse = count_code_elements(x)
    ast_parse['nested_counts'] = {'nested_' + k: val for k, val in ast_parse['nested_counts'].items()}
    #ast_parse['counts'].pop('FunctionDef')
    # Regrouping everything together and dividing by LOC to normalize data
    ast_parse = {k: ast_parse['counts'].get(k, 0) + ast_parse['nested_counts'].get(k, 0) for k in set(ast_parse['counts']) | set(ast_parse['nested_counts'])} 
    ast_parse['loc'] = loc

    return ast_parse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=None)
    parser.add_argument('-t', '--threshold', default=0.5, type=float)
    args = parser.parse_args()

    threshold = float(args.threshold)

    if args.dataset == 'humanevalplus':
        task_nb = 164
    elif args.dataset == 'ClassEval':
        task_nb = 200
        sampled_task = pd.read_csv(os.path.join('..', 'data', 'classeval','sampled_tasks.csv'), sep=';')
    else:
        raise Exception(f"Dataset {args.dataset} not recognised")

    eval_list, sim_list = [], []
    models_list = ['deepseek', 'magicoder', 'llama', 'gpt', 'gemma', 'deepseek-chat', 'qwen-coder', 'yi-coder']
    for model in models_list:
        with open(os.path.join('..', 'data', args.dataset.lower(), 'post_test', f'results_{args.dataset}_{model}_eval.json'), 'r') as f:
                eval_list.append(json.load(f))
        with open(os.path.join('..', 'data', args.dataset.lower(), 'sim', f'results_{args.dataset}_{model}_sim.json'), 'r') as f:
                sim_list.append(json.load(f))

    code_dict = {}
    oracle_codes = pd.read_csv(os.path.join('..', 'data', args.dataset.lower(), f'prompts_generated_{args.dataset}.csv'), sep=';')
    oracle_codes['class_id'] = oracle_codes['class_id'].astype(str) # making sure it works for all benchmarks

    for ind_key in tqdm.tqdm(eval_list[0].keys()):
        code_dict[ind_key] = {}
        for mod, eval_l in enumerate(eval_list):    
            code_dict[ind_key][models_list[mod]] = []
            for key in eval_l[str(ind_key)]:
                if key != 'original prompt':
                    for ind_seed in range(len(eval_l[str(ind_key)][key])):
                        sim_ = sim_list[mod][str(oracle_codes.index[oracle_codes['class_id'] == str(ind_key).split('ClassEval_')[-1]].values[0])][key][ind_seed]
                        
                        if sim_ > threshold:                        
                            try:
                                loc = analyze(eval_l[str(ind_key)][key][ind_seed][0]).sloc # removing comments/blank etc.
                                ast_parse = ast_parse_f(eval_l[str(ind_key)][key][ind_seed][0], loc)
                                code_dict[ind_key][models_list[mod]].append((eval_l[str(ind_key)][key][ind_seed][0], ast_parse))
                            # If the code does not compile, skip it
                            except Exception as e:
                                continue

    with open(os.path.join(os.path.join('..', 'data', args.dataset.lower(), f'structure_types_{args.dataset}_{args.threshold}.json')), 'w') as f:
        json.dump(code_dict, f)

