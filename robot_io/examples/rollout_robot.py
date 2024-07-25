import hydra

target_pos = []
target_orn = []
gripper_action = 1
target_action = {'motion': (target_pos, target_orn, gripper_action),  'ref': 'abs'}

@hydra.main(config_path="../conf", config_name="rollout_trajectory")
def main(cfg):
    """
    Rollout and save a trajectory, with default controllers.

    Args:
        cfg: Hydra config
    """
    recorder = hydra.utils.instantiate(cfg.recorder)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    obs = env.reset()

    for i in range(100):
        # get_action() # linearize....
        next_obs, rew, done, info = env.step(target_action)
        print(robot.get_state())
        recorder.step(action, obs, rew, done, info)
        env.render()
        obs = next_obs
    print('Rollout done for 100 steps.')


if __name__ == "__main__":
    main()
