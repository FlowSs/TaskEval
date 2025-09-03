import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from openai import OpenAI

import tqdm
import torch
import pandas as pd
import argparse
import json

import time

from const import *
from const_class import *

corresponding_table = {
   'llama': 'codellama/CodeLlama-7b-Instruct-hf',
   'gemma': 'google/codegemma-7b-it',
   'magicoder': 'ise-uiuc/Magicoder-S-DS-6.7B',
   'deepseek': 'deepseek-ai/deepseek-coder-6.7b-instruct',
   'gpt': 'gpt-3.5-turbo-0125',
   'deepseek-chat': 'deepseek-chat',
   'qwen-coder': 'Qwen/Qwen2.5-Coder-7B-Instruct',
   'yi-coder': '01-ai/Yi-Coder-1.5B-Chat',
}

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', '--model', default=None)
   parser.add_argument('-d', '--dataset', default=None)
   args = parser.parse_args()

   if args.dataset not in ['humanevalplus', 'ClassEval']:
      raise Exception(f"Dataset {args.dataset} not recognised")
   
   if args.model not in corresponding_table:
      raise Exception(f"Model {args.model} not recognised")

   # Needs a file name 'key.json' to store the API keys for OpenAI
   if not(os.path.exists('key.json')):
    raise Exception(f'You need to provide a key.json file containing the OpenAI key in order to use GPT4')
     
   with open('key.json', 'r') as f:
    keys = json.load(f)
   
   model_name = corresponding_table[args.model]
   if 'gpt' == args.model:
      client = OpenAI(api_key=keys['OPENAI'])
   elif 'deepseek-chat' == args.model:
      client = OpenAI(api_key=keys['DEEPSEEK'], base_url="https://api.deepseek.com")
   else:
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=keys['HF_TOKEN'])
      model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', token=keys['HF_TOKEN'], trust_remote_code=True)
      model = torch.compile(model)
      model.eval()

   file_data = f'prompts_generated_{args.dataset}.csv'
   dat = pd.read_csv(os.path.join('..', 'data', f'{args.dataset.lower()}', file_data), \
                     sep=";", index_col=0 if args.dataset == 'humanevalplus' else None)

   if 'magicoder' == args.model:
      prompt_base = prompt_magic if args.dataset =='humanevalplus' else prompt_magic_c
   elif args.model in ['deepseek-chat', 'deepseek', 'qwen-coder', 'yi-coder']:
      prompt_base = prompt_deepseek if args.dataset =='humanevalplus' else prompt_deepseek_c
   elif 'llama' == args.model:
      prompt_base = prompt_codellama if args.dataset =='humanevalplus' else prompt_codellama_c
   elif 'gpt' == args.model:
      prompt_base = prompt_gpt if args.dataset =='humanevalplus' else prompt_gpt_c
   elif 'gemma' == args.model:
      prompt_base = prompt_gemma if args.dataset =='humanevalplus' else prompt_gemma_c

   file_output = f'results_{args.dataset}_{args.model}_just_one.json'
   print(f"Running on dataset: {args.dataset} and model: {model_name}")

   if os.path.exists(os.path.join('..', 'data', f'{args.dataset.lower()}', 'raw', file_output)):
      with open(os.path.join('..', 'data', f'{args.dataset.lower()}', 'raw', file_output), 'r') as f:
         all_tasks = json.load(f)
   else:
      all_tasks = {}

   for task_id in tqdm.tqdm(dat.index.to_numpy()):
        # If we already have the task
        if str(task_id) in all_tasks:
           print('Skipped')
           continue
        
        all_tasks[str(task_id)] = {}

        # Setting base prompt, common for a task across level
        if args.dataset == 'humanevalplus':
          signature = dat.loc[task_id]['signature']
          prompt_base_plus = prompt_base.replace('CODE_PLACEHOLDER', signature)
        elif args.dataset == 'ClassEval':
          class_name = dat.loc[task_id]['class_name']
          method_name = dat.loc[task_id]['method_name']
          class_text = dat.loc[task_id]['class_text']
          method_param = dat.loc[task_id]['method_params']
          signature = dat.loc[task_id]['method_signature']
          
          prompt_base_plus = prompt_base.replace('METHOD_NAME', method_name)
          prompt_base_plus = prompt_base_plus.replace('CLASS_NAME', class_name)
          prompt_base_plus = prompt_base_plus.replace('CLASS_CODE', class_text.strip())
          prompt_base_plus = prompt_base_plus.replace('METHOD_SIGNATURE', signature.strip())
        
        prompt_list = {}
        for type, prompt in zip(['level 1', 'level 2',
          'level 3'], dat.loc[task_id][['level 1', 'level 2',
          'level 3']]):
            try:
              list_prompt = eval(prompt)
              prompt_list[type] = []
              for p in list_prompt:
                 prompt_list[type].append(p)
            except:
              prompt_list[type] = [prompt]

        # For each level (1, 2 and 3)
        for key in prompt_list.keys():
            all_tasks[str(task_id)][key] = []
            for prompt in prompt_list[key]:
                # Adding the docstring/instruction
                if args.dataset == 'humanevalplus':
                   input_prompt = prompt_base_plus.replace('INSTRUCTION_PLACEHOLDER', prompt)
                elif args.dataset == 'ClassEval':
                   # Get the indentation of the code
                   indent = class_text.split('pass')[-2].split(":\n")[-1]
                   prompt_ = f"\n{indent}" + f"\n{indent}".join(prompt.split("\n"))
                   if key == 'original' or method_param.strip() == '':
                     to_replace = indent + '\"\"\"' + prompt_ + '\n' + indent + '\"\"\"'
                   else:
                     to_replace = indent + '\"\"\"' + prompt_ + '\n' + indent + f"\n{indent}".join(method_param.split("\n")) + '\"\"\"'
                   input_prompt = prompt_base_plus.replace('DOCSTRING', to_replace).strip()

                # Using HF implementation
                if args.model not in ['gpt', 'deepseek-chat']:
                   tokenized_text = tokenizer([input_prompt], return_tensors='pt').to(model.device)
                   for i in range(5):
                    set_seed(i)
                    output = model.generate(**tokenized_text,\
                                        max_new_tokens=1024,
                                        do_sample=True,
                                        temperature=0.8,
                                        pad_token_id = tokenizer.eos_token_id
                                      )
                    print(tokenizer.decode(output[0])) 
                    all_tasks[str(task_id)][key].append(tokenizer.decode(output[0]))
                # Otherwise, for GPT3.5
                else:
                   inference_not_done = True
                   number_of_trials = 0
                   error_message = []

                   # While inference is not completed or if we haven't tried at least 3 times (i.e. to avoid server error)
                   while inference_not_done and number_of_trials != 3:
                     try:
                      if 'gpt' == args.model:
                       completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                         {"role": "user", "content": input_prompt}
                        ],
                        max_tokens=1024,
                        temperature=0.8,
                        seed=0,
                        n=5,
                       )
                       out = [completion.choices[k].message.content for k in range(len(completion.choices))]
                      else:
                        out = []
                        for i in range(5):
                          completion = client.chat.completions.create(
                           model=model_name,
                           messages=[
                             {"role": "user", "content": input_prompt}
                           ],
                           max_tokens=1024,
                           temperature=0.8,
                           seed=i,
                           stream=False,
                          )
                          out.append(completion.choices[0].message.content)
                      inference_not_done = False
                     # in case something wrong happen, retry
                     except Exception as e:
                      error_message.append(e)
                      time.sleep(2)
                      number_of_trials += 1
                   
                   all_tasks[str(task_id)][key].extend(out)

        with open(os.path.join('..', 'data', f'{args.dataset.lower()}', 'raw', file_output), 'w') as f:
             json.dump(all_tasks, f)

