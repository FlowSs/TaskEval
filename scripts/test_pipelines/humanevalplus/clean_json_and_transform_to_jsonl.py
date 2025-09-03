import os
import json
from evalplus.data import write_jsonl
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None)
parser.add_argument('-d', '--dataset', default=None)
args = parser.parse_args()

with open(os.path.join('..', '..', '..', 'data', 'humanevalplus', 'raw', f'results_humanevalplus_{args.model}.json'), 'r') as f:
  dat = json.load(f)

reg_str = "```python((.|\n)*?)```" if 'llama' not in args.model else "\[PYTHON\]((.|\n)*?)\[/PYTHON\]"

if args.dataset == 'humanevalplus':
  
    samples = []
    for key in dat:
      for k in dat[key]:
        for ind, sol in enumerate(dat[key][k]):
          samples.append(dict(task_id=f'HumanEval/{int(key)}'))
          samples[-1]['prompt_type'] = f"{k}_{ind}"
          code = re.findall(reg_str, sol) 
          if len(code) < 2 and args.model not in ['gpt', 'deepseek-chat']:
            samples[-1]['solution'] = ''
            continue
          elif args.model in ['gpt', 'deepseek-chat'] and len(code) == 0:
            samples[-1]['solution'] = ''
            continue
          
          if args.model not in ['gpt', 'deepseek-chat']:
             raw_code = code[1][0]
          else:
             raw_code = code[0][0]
          screened_lines = []
          lines = raw_code.split('\n')
          for line in lines:
            if line.startswith('assert') or line.startswith("#") or line.startswith("print"):
             continue
            screened_lines.append(line)
              
          processed_code = '\n'.join(screened_lines)
          samples[-1]['solution'] = processed_code

    # for i in range(164):
    #     if i != 126:
    #        samples.append(dict(task_id=f'HumanEval/{i}'))
    #        samples[-1]['solution'] = ''

    write_jsonl(os.path.join('..', '..', '..', 'data', 'humanevalplus', 'raw', f'results_humanevalplus_{args.model}.jsonl'), samples)
else:
  raise NotImplementedError
