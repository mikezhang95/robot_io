
import os
import hydra
import time
import numpy as np
import torch
from robot_io.utils.utils import depth_img_from_uint16, quat_to_euler, euler_to_quat, to_relative_all_frames
current_dir = os.path.dirname(os.path.abspath(__file__))


AGENT_TYPE = 'llmpc'
EPISODE_LENGTH = 50
TARGET_POS = np.array([0.35, 0.1, 0.2]) # test data
# TARGET_POS = np.array([0.1942679 , 0.49881903, 0.52366607]) # blue light


def calculate_reward(obs):
    current_pos = obs['robot_state']['tcp_pos']
    cost = np.sqrt(np.sum((current_pos-TARGET_POS)**2))
    return -cost

def processed_obs(obs):
    """ convert dict to np.array
    """
    robot_state = obs["robot_state"]
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
        print('[Warning] action is clipped under relative movement of 0.1.')
    tcp_pos = np.clip(tcp_pos - obs[:3], -0.05, 0.05) + obs[:3]
    tcp_orn = [1, 0, 0, 0]
    gripper_action = 1
    target_action = {'motion': (tcp_pos, tcp_orn, gripper_action),  'ref': 'abs'} # todo: always open gripper
    return target_action


@hydra.main(config_path="../conf", config_name="rollout_trajectory")
def main(cfg):
    """
    Rollout and save a trajectory, with different controllers.

    Args:
        cfg: Hydra config
    """

    # initialize environment
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    recorder = hydra.utils.instantiate(cfg.recorder, env=env, save_dir=cfg.save_dir)

    # initialize agents
    nq, nv, nu = 10, 0, 3
    if AGENT_TYPE == 'llmpc':
        from model import LatentLinearModel 
        from agent import LLMController
        model = LatentLinearModel(nq+nv, nu,  {'hidden_dim': 10, 'dynamic_structure': 'companion', 'cost_structure': 'psd_monotonic'})
        model_path = f'{current_dir}/runs/panda_real-latent_linear-move_test/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = LLMController(model, model_path)

    elif AGENT_TYPE == 'il':
        from model import PolicyModel 
        from agent import ILController 
        model = PolicyModel(nq+nv, nu, {'hidden_dim': 100})
        model_path = f'{current_dir}/runs/panda_real-policy-move_test/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = ILController(model)

    elif AGENT_TYPE == 'traj':
        from agent import TrajController 
        data_path = f'{current_dir}/replays/move_test'
        start_end_ids = np.sort(np.load(os.path.join(data_path, "ep_start_end_ids.npy")), axis=0)[0]
        agent = TrajController(data_path, start_end_ids)
        robot_state = agent.init_robot_state
        gripper_state = "open" if robot_state["gripper_opening_width"] > 0.07 else "closed"
        
    elif AGENT_TYPE == 'expert':
        raise 
    
    # reset environment
    if AGENT_TYPE == 'traj':
        obs = env.reset(
            target_pos=robot_state["tcp_pos"],
            target_orn=robot_state["tcp_orn"],
            gripper_state=gripper_state,
        )
        episode_length = start_end_ids[1] - start_end_ids[0] + 1
    else:
        obs = env.reset()
        episode_length = EPISODE_LENGTH

    # start to rollout! 
    for i in range(episode_length):
        # processed_obs
        obs_array = processed_obs(obs)
        # === get action ===
        action_array = agent.act(obs_array, step=i)
        # processed action
        action = processed_action(action_array, obs_array)
        # === step env ===
        next_obs, reward, done, info = env.step(action)
        # calculate reward
        reward = calculate_reward(next_obs)
        if i == episode_length-1: done = True
        print(i, obs_array[:3], action, reward, '\n')
        # TODO: record data
        recorder.step(action, obs, reward, done, info)
        obs = next_obs
    print('- Episode ends!')


if __name__ == "__main__":
    main()

