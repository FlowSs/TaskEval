import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'birt-gd', 'src'))

from birt import Beta3


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', '--dataset', default=None)
   args = parser.parse_args()
   
   if args.dataset not in ['CE', 'HE']:
      raise Exception('Dataset should be CE (ClassEval) or HE (HumanEval)')

   data = np.load(os.path.join('..', 'data', 'irt_data', 'test_data_all.npy'))
   data = data[:164] if args.dataset == 'HE' else data[164:]
   b3 = Beta3(n_respondents=data.shape[1], n_items=data.shape[0], epochs=20000, n_inits=3000, random_seed=0, learning_rate=1, n_workers=-1, tol=1e-7)
   b3.fit(data)
   print(b3.score)
   print(b3.summary())

   np.save(os.path.join('..', 'data', 'irt_data', f'diff_all_tasks_test_{args.dataset}.npy'), b3.difficulties)
   np.save(os.path.join('..', 'data', 'irt_data', f'disc_all_tasks_test_{args.dataset}.npy'), b3.discriminations)
   np.save(os.path.join('..', 'data', 'irt_data', f'ab_all_tasks_test_{args.dataset}.npy'), b3.abilities)
