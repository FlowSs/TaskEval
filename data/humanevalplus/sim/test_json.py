import json

for model in ['deepseek', 'gemma', 'gpt', 'llama', 'magicoder']:
  with open(f'results_humanevalplus_{model}_sim.json') as f:
     dat = json.load(f)
  with open(f'old/results_humanevalplus_{model}_sim.json') as f:
     dat2 = json.load(f)

  diff = 0
  for key in dat:
    for level in dat[key]:
       for (ele, ele2) in zip(dat[key][level], dat2[key][level]):
         if ele != ele2:
            diff += 1
         if ele < ele2:
            #print("ELE < ELE2: ", key, level, ele, ele2)
            continue
            #raise Exception("Something is wrong...")

  print(model, diff/(200*90)*100, diff)
