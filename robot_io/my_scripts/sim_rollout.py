
import os
import hydra
import time
import numpy as np
import imageio
import torch
from robot_io.utils.utils import depth_img_from_uint16, quat_to_euler, euler_to_quat, to_relative_all_frames
current_dir = os.path.dirname(os.path.abspath(__file__))

from utils import *


AGENT_TYPE = 'traj'
EPISODE_LENGTH = 50
NUM_EPISODES = 1

TARGET_POS = np.array([0.35, 0.05, 0.45]) # test data
# TARGET_POS = np.array([0.18 , 0.18, 0.55]) # blue light

@hydra.main(config_path="../calvin_env/conf", config_name="config_data_collection")
def main(cfg):
    """
    Rollout and save a trajectory, with different controllers.

    Args:
        cfg: Hydra config
    """

    # initialize environment
    # M: no need to initialize robot
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
    recorder = hydra.utils.instantiate(cfg.recorder, env=env, save_dir=cfg.save_dir)

    # initialize agents
    nq, nv, nu = 10, 0, 3
    if AGENT_TYPE == 'llmpc':
        from model import LatentLinearModel 
        from agent import LLMController
        model = LatentLinearModel(nq+nv, nu,  {'hidden_dim': 10, 'dynamic_structure': 'companion', 'cost_structure': 'psd_monotonic'})
        model_path = f'{current_dir}/runs/panda_real-latent_linear-move_test/model_last'
        # model_path = f'{current_dir}/runs/panda_real-latent_linear-blue_light/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = LLMController(model, model_path)

    elif AGENT_TYPE == 'il':
        from model import PolicyModel 
        from agent import ILController 
        model = PolicyModel(nq+nv, nu, {'hidden_dim': 100})
        model_path = f'{current_dir}/runs/panda_real-policy-move_test/model_last'
        # model_path = f'{current_dir}/runs/panda_real-policy-blue_light/model_last'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.to(device)
        print(f'Model Loaded from {model_path} to device {model.device}')
        agent = ILController(model)

    elif AGENT_TYPE == 'traj':
        from agent import TrajController 
        data_path = f'{current_dir}/replays/move_test_rerun'
        # data_path = f'{current_dir}/replays/blue_light_raw'
        start_end_ids = np.sort(np.load(os.path.join(data_path, "ep_start_end_ids.npy")), axis=0)[0]
        start_end_ids = [0, 49]
        agent = TrajController(data_path, start_end_ids)
        robot_state = agent.init_robot_state
        gripper_state = "open" if robot_state["gripper_opening_width"] > 0.07 else "closed"
        print(start_end_ids)
        
    elif AGENT_TYPE == 'expert':
        raise 
    
    frames = []
    for j in range(NUM_EPISODES):

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
            frames.append(env.render(mode='rgb_array'))
            # processed_obs
            obs_array = processed_obs(obs)
            # === get action ===
            action_array = agent.act(obs_array, step=i)
            # processed action
            action = processed_action(action_array, obs_array)
            # === step env ===
            next_obs, reward, done, info = env.step(action)
            # calculate reward
            reward = calculate_reach_reward(obs_array[:3], TARGET_POS)
            if i == episode_length-1: done = True
            # record data
            recorder.step(action, obs, reward, done, info)
            obs = next_obs
            print(i, obs_array[:3], action, reward, '\n')
        

        print(f'- Episode {j+1} ends with {i+1} transitions!\n')
 
    # save video
    fps = env.control_freq
    video_path = "rollout.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    print(f'Video saved in {video_path} with fps:{fps}\n')

if __name__ == "__main__":
    main()

