import json
import os
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None)
args = parser.parse_args()

with open(os.path.join('..', '..', '..', 'data', 'humanevalplus', 'raw', f'results_humanevalplus_{args.model}_results.jsonl'), 'r') as f:
	eval_ = json.load(f)

with open(os.path.join('..', '..', '..', 'data', 'humanevalplus', 'raw', f'results_humanevalplus_{args.model}.json'), 'r') as f:
	code = json.load(f)

code_copy = copy.deepcopy(code)
code_copy_base = copy.deepcopy(code)

for task in eval_['eval']:
  task_id = task.split('/')[1]
  
  for sample, s in enumerate(eval_['eval'][task]):
    if 0 <= sample <= 29:
      key = 'level 1'
      samp_id = sample
    elif 30 <= sample <= 59:
      key = 'level 2'
      samp_id = sample - 30
    else:
      key = 'level 3'
      samp_id = sample - 60

    code_copy[task_id][key][samp_id] = (eval_['eval'][task][sample]['solution'], True if eval_['eval'][task][sample]['plus_status'] != 'fail' else False)

with open(os.path.join('..', '..', '..', 'data', 'humanevalplus', 'post_test', f'results_humanevalplus_{args.model}.json'), 'w') as f:
	json.dump(code_copy, f)

