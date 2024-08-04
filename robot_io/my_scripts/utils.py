
import time
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir + '/..')
sys.path.append(current_dir)
sys.path.append(parent_dir)

import numpy as np
import torch
import random

from scipy.spatial.transform.rotation import Rotation as R

def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")

def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()

def calculate_reward(current_pos, target_pos):
    cost = np.sqrt(np.sum((current_pos-target_pos)**2))
    return -cost


def processed_obs(robot_state):
    """ convert dict to np.array
    """
    tcp_pos = robot_state["tcp_pos"]
    tcp_orn = robot_state["tcp_orn"]
    gripper_width = robot_state["gripper_opening_width"]
    joint_positions = robot_state["joint_positions"]
    gripper_action = 1.0 # todo: always open gripper

    # M: now only save [tcp_pos, joint_positions] as state, 10-d
    robot_obs = np.concatenate([tcp_pos, joint_positions])
    # robot_obs = np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])
    return robot_obs


def processed_action(action, obs):
    """ convert np.array to dict
    """
    if type(action) == dict: # coming from frame file
        tcp_pos, tcp_orn, gripper_action = action['motion']
    else:
        tcp_pos = action[:3]
        # M: only use tcp_pos as action
        tcp_orn = [1, 0, 0, 0]
        grippoer_action = 1.0
        # tcp_orn = action[3:-1]
        # tcp_orn = euler_to_quat(tcp_orn)if len(tcp_orn) == 3 else tcp_orn
        # gripper_action = action[-1]


    # TODO: CHECK ACTIONS
    if np.max(np.abs(tcp_pos - obs[:3])) > 0.05:
        print('[Warning] action is clipped under relative movement of 0.05.')
    tcp_pos = np.clip(tcp_pos - obs[:3], -0.05, 0.05) + obs[:3]
    tcp_orn = [1, 0, 0, 0]
    gripper_action = 1
    target_action = {'motion': (tcp_pos, tcp_orn, gripper_action),  'ref': 'abs'} # todo: always open gripper
    return target_action


def serialize_action(action):
    """ convert np.array to dict
    """
    tcp_pos, tcp_orn, gripper_action = action['motion']
    # M: now only save [tcp_pos] as action now, 3-d
    # action_array = np.concatenate([tcp_pos, quat_to_euler(tcp_orn), gripper_action]) # TODO: should use euler as action
    action_array = tcp_pos
    return action_array


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def np_to_torch(t, device='cpu'):
    if t is None:
        return None
    else:
        return torch.Tensor(t).to(device)

def torch_to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
