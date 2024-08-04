
import os, sys
import numpy as np
import pickle

from utils import *


TARGET_POS = np.array([0.35, 0.05, 0.45])
# TARGET_POS = np.array([0.18, 0.18, 0.55])

current_dir = os.path.dirname(os.path.abspath(__file__))
l = 50
dataset_dir = f'{current_dir}/replays/move_test_rerun'
output_path = f'{current_dir}/panda-real_l50_e100.pkl'
# l = 77
# dataset_dir = f'{current_dir}/replays/blue_light_rerun'
# output_path = f'{current_dir}/panda-real_l100_e100.pkl'
                       

# 1. load episode ids...
start_end_ids = np.sort(np.load(os.path.join(dataset_dir, "ep_start_end_ids.npy")), axis=-1)
ep_start_end_ids = [[i*l, (i+1)*l-1] for i in range(100)]

# 2. create dataset
all_data = []
for push_id in ep_start_end_ids:
    print(push_id)
    observations, actions, costs = [], [], []
    for step in range(push_id[0], push_id[1]+1):
        file_path = os.path.join(dataset_dir, f'frame_{step:06d}.npz')
        raw_data = np.load(file_path, allow_pickle=True)
       
        # process data
        obs_array = processed_obs(raw_data['robot_state'].item())
        action_array = serialize_action(raw_data['action'].item())
       
        # save data
        observations.append(obs_array)
        actions.append(action_array) 
        costs.append(-calculate_reacy_rewards(obs_array[:3], TARGET_POS))
          

    observations = np.array(observations)
    actions = np.array(actions)
    costs = np.array(costs)

    print(observations.shape)
    
    data = {'observations': observations[:-1],
            'next_observations': observations[1:], 
            'actions': actions[:-1], 
            'rewards': -costs[:-1], 
            'costs': np.expand_dims(costs[:-1], axis=-1), 
            'time_costs': 0.0*costs[:-1]}
    all_data.append(data)

with open(output_path, 'wb') as wf:
    pickle.dump(all_data, wf)
print(f'Save {len(all_data)} in {output_path}.\n')
      
      
      
