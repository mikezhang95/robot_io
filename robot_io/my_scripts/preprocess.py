


import os, sys
import numpy as np
import pickle
from scipy.spatial.transform.rotation import Rotation as R


TARGET_POS = np.array([0.35, 0.05, 0.45])
# TARGET_POS = np.array([0.18, 0.18, 0.55])


current_dir = os.path.dirname(os.path.abspath(__file__))
l = 50
dataset_dir = f'{current_dir}/replays/move_test_rerun'
output_path = f'{current_dir}/panda-real_l50_e100.pkl'
# l = 77
# dataset_dir = f'{current_dir}/replays/blue_light_rerun'
# output_path = f'{current_dir}/panda-real_l100_e100.pkl'


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")

def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()

def calculate_reward(obs):                    
    current_pos = obs[:3]                     
    cost = np.sqrt(np.sum((current_pos-TARGET_POS)**2))
    return -cost                              
                  
    
def processed_obs(robot_state):
    """ convert dict to np.array
    """
    tcp_pos = robot_state["tcp_pos"]
    tcp_orn = robot_state["tcp_orn"] # TODO: should use quat as state
    gripper_width = robot_state["gripper_opening_width"]
    joint_positions = robot_state["joint_positions"]
    gripper_action = 1.0 # todo: always open gripper
    
    # M: now only save [tcp_pos, joint_positions] as state, 10-d
    # robot_obs = np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])
    robot_obs = np.concatenate([tcp_pos, joint_positions])
    return robot_obs


def processed_action(action):
    """ convert np.array to dict
    """
    tcp_pos, tcp_orn, gripper_action = action['motion']
    # M: now only save [tcp_pos] as action now, 3-d
    # action_array = np.concatenate([tcp_pos, quat_to_euler(tcp_orn), gripper_action]) # TODO: should use euler as action
    action_array = tcp_pos
    return action_array


# 1. load episode ids...
ep_start_end_ids = np.load(os.path.join(dataset_dir, 'ep_start_end_ids.npy'))
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
        action_array = processed_action(raw_data['action'].item())
       
        # save data
        observations.append(obs_array)
        actions.append(action_array) 
        costs.append((np.sqrt(np.sum(obs_array[:3]-TARGET_POS)**2)))

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
      
      
      
