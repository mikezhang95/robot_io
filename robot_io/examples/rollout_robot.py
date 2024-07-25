import hydra
import time

target_pos = [0.35, 0.1, 0.2]
target_orn = [1.0, 0.0, 0.0, 0.0]
gripper_action = 1
target_action = {'motion': [target_pos, target_orn, gripper_action],  'ref': 'abs'}
action = target_action


@hydra.main(config_path="../conf", config_name="rollout_trajectory")
def main(cfg):
    """
    Rollout and save a trajectory, with default controllers.

    Args:
        cfg: Hydra config
    """
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    recorder = hydra.utils.instantiate(cfg.recorder, env=env, save_dir=cfg.save_dir)

    obs = env.reset()

    for i in range(100):
        # get_action() # linearize....
        current_pos = obs['robot_state']['tcp_pos']
        truncated_pos = ( target_pos - current_pos ) / max(100-i, 1)  + current_pos
        action['motion'][0] = truncated_pos 
        t0 = time.time()
        next_obs, rew, done, info = env.step(action)
        t1 = time.time()
        print(i, robot.get_state(), t1-t0)
        recorder.step(action, obs, rew, done, info)
        obs = next_obs
    print('Rollout done for 100 steps.')


if __name__ == "__main__":
    main()
