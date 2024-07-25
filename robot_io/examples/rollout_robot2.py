import hydra
import numpy as np
import time
import pickle

@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):
    """
    Starting from the neutral position, move the EE left and right.

    Args:
        cfg: Hydra config.
    """
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # left_pos = pos + np.array([0, -0.1, 0])
    # right_pos = pos + np.array([0, 0.1, 0])
    # left_pos = pos + np.array([-0.05, 0., 0])
    # right_pos = pos + np.array([0.05, 0., 0])
    # left_pos = pos + np.array([0.05, 0.1, -0.4])
    # right_pos = pos + np.array([0., 0., 0.0])

    # robot.move_cart_pos_abs_ptp(target_pos, orn)
    data_all = []
    for j in range(20):
        robot.move_to_neutral()
        pos, orn = robot.get_tcp_pos_orn()
        for i in range(50):
            target_pos = pos + np.array([0.05, 0.05, -0.2])/50.0 * (i+1)
            robot.move_async_cart_pos_abs_ptp(target_pos, orn)
            time.sleep(0.033)
            print(i, env._get_obs()['robot_state']['tcp_pos'], '\n')
            data_all.append({'target_pos': target_pos, 'obs': env._get_obs()})

    with open('/home/administrator/yuan_panda/robot_io/robot_io/examples/test.pickle', 'wb') as wf:
        pickle.dump(data_all, wf)
    print('saved')


if __name__ == "__main__":
    main()
