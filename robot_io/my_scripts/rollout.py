
import hydra
import time
import numpy as np
import torch
from robot_io.utils.utils import depth_img_from_uint16, quat_to_euler, euler_to_quat, to_relative_all_frames


AGENT_TYPE = 'llmpc'
EPISODE_LENGTH = 50
TARGET_POS = np.array([0.35, 0.1, 0.2]) # test data
TARGET_POS = np.array([0.1942679 , 0.49881903, 0.52366607]) # blue light


def calculate_reward(obs):
    current_pos = obs[:3]
    cost = np.sqrt(np.sum((current_pos-TARGET_POS)**2))
    return -cost

def processed_obs(obs):
    """ convert dict to np.array
    """
    robot_state = obs["robot_state"]
    tcp_pos = robot_state["tcp_pos"]
    tcp_orn = quat_to_euler(robot_state["tcp_orn"])
    gripper_width = robot_state["gripper_opening_width"]
    joint_positions = robot_state["joint_positions"]
    gripper_action = 1.0 # todo: always open gripper
    robot_obs = np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])
    return robot_obs


def processed_action(action, obs):
    """ convert np.array to dict
    """
    # check large movement
    tcp_pos = action[:3]
    tcp_orn = action[3:-1]
    gripper_action = action[-1]
    tcp_orn = euler_to_quat(tcp_orn)if len(tcp_orn) == 3 else tcp_orn

    # CHECK ACTIONS
    if np.max(np.abs(tcp_pos - obs[:3])) > 0.1:
        print('[Warning] action is clipped under relative movement of 0.1.')
    tcp_pos = np.clip(tcp_pos - obs[:3], -0.1, 0.1) + obs[:3]
    tcp_orn = [1, 0, 0, 0]
    gripper_action = 1
    obs[:3]
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
    nq, nv, nu = 15, 0, 7
    if AGENT_TYPE == 'llmpc':
        from model import LatentLinearModel 
        from agent import LLMController
        model = LatentLinearModel(nq+nv, nu,  {'hidden_dim': 10, 'dynamic_structure': 'companion', 'cost_structure': 'psd_monotonic'})
        model_path = f'{parent_dir}/outputs/runs/humanoid_stand-latent_linear2-h10_nz/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = LLMController(model)

    elif AGENT_TYPE == 'il':
        from model import PolicyModel 
        from agent import ILController 
        model = PolicyModel(nq+nv, nu, {'hidden_dim': 210})
        model_path = f'{parent_dir}/outputs/runs/humanoid_stand-policy-h210_new2/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = ILController(model)

    elif AGENT_TYPE == 'traj':
        from agent import TrajController 
        data_path = 'taco_replay'
        start_end_ids = np.sort(np.load(os.path.join(data_path, "ep_start_end_ids.npy")), axis=0)[0]
        agent = TrajController(data_path, start_end_ids)
        robot_state = agent.init_robot_state
        gripper_state = "open" if robot_state["gripper_opening_width"] > 0.07 else "closed"
        
        
    elif AGENT_TYPE == 'expert':
        raise 
    
    # reset environment
    if AGENT_TYPE == 'traj':
        env.reset(
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
        obs = processed_obs(obs)
        # === get action ===
        action = agent.act(obs, step=i)
        # processed action
        action = processed_action(action, obs)
        # === step env ===
        next_obs, reward, done, info = env.step(action)
        # calculate reward
        reward = calculate_reward(next_obs)
        print(i, obs, action, reward)
        # TODO: record data
        recorder.step(action, obs, reward, done, info)
        obs = next_obs



if __name__ == "__main__":
    main()

