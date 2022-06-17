import argparse

import numpy as np
from smac.env import StarCraft2Env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map', type=str,
                        choices=['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '27m_vs_30m',
                                 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '3s_vs_4z',
                                 '3s_vs_5z', '1c3s5z', '2m_vs_1z',
                                 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane',
                                 '2c_vs_64zg', ])
    parser.add_argument('--n_episodes', type=int, default=10, help='total episode num of training process')

    args = parser.parse_args()

    env = StarCraft2Env(map_name=args.map)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    steps_per_episode = env_info["episode_limit"]
    print(env_info)

    for cur_episode in range(args.n_episodes):
        env.reset()
        terminated = False
        step = 0  # record the length of this episode
        episode_reward = 0
        # save the trajectory of this episode

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering
            # print(obs)
            # print(state)

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward
            step += 1

        print(f"episode: {cur_episode + 1}, step: {step}, episode reward: {episode_reward}")
    env.close()
