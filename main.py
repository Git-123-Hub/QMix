import argparse

import numpy as np
from smac.env import StarCraft2Env

from qmix import QMix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map', type=str,
                        choices=['3m', '8m', '25m', '5m_vs_6m', '8m_vs_9m', '10m_vs_11m', '27m_vs_30m',
                                 'MMM', 'MMM2', '2s3z', '3s5z', '3s5z_vs_3s6z', '3s_vs_3z', '3s_vs_4z',
                                 '3s_vs_5z', '1c3s5z', '2m_vs_1z',
                                 'corridor', '6h_vs_8z', '2s_vs_1sc', 'so_many_baneling', 'bane_vs_bane',
                                 '2c_vs_64zg', ])
    parser.add_argument('--n_episodes', type=int, default=10, help='total episode num of training process')
    parser.add_argument('--capacity', type=int, default=5e3, help='maximum number of episode in buffer')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of episode sampled each time from buffer')

    args = parser.parse_args()

    env = StarCraft2Env(map_name=args.map)
    env_info = env.get_env_info()
    state_dim = env_info["state_shape"]
    obs_dim = env_info["obs_shape"]
    action_dim = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    steps_per_episode = env_info["episode_limit"]

    qmix = QMix(n_agents, state_dim, obs_dim, action_dim, args.capacity, steps_per_episode)

    for cur_episode in range(args.n_episodes):
        env.reset()
        terminate = False
        step = 0  # record the length of this episode
        episode_reward = 0
        # save the trajectory of this episode
        obs_list, state_list, action_list, avail_action_list, reward_list, terminate_list = [], [], [], [], [], []

        while not terminate:  # interact with the env for an episode
            obs = env.get_obs()
            state = env.get_state()
            # record last action of each agent for choosing action
            last_actions = np.zeros(n_agents, action_dim)
            # env.render()  # Uncomment for rendering

            avail_actions = [env.get_avail_agent_actions(agent_id) for agent_id in range(n_agents)]
            # todo: add args for epsilon
            actions = qmix.choose_action(obs, last_actions, avail_actions, 0.1)
            reward, terminate, _ = env.step(actions)
            # todo: record win info
            episode_reward += reward
            step += 1
            # save experience
            obs_list.append(obs)
            state_list.append(state)
            action_list.append(actions)
            avail_action_list.append(avail_actions)
            reward_list.append(reward)
            terminate_list.append(terminate)

            last_actions = actions  # update last action

        # after an episode finishes, we have to save the last data and then save them all
        avail_actions = []
        for agent_id in range(n_agents):
            avail_actions.append(env.get_avail_agent_actions(agent_id))
        obs_list.append(env.get_obs())
        state_list.append(env.get_state())
        avail_action_list.append(avail_actions)
        qmix.buffer.add(obs_list, state_list, action_list, avail_action_list, reward_list, terminate_list)

        print(f"episode: {cur_episode + 1}, step: {step}, episode reward: {episode_reward}")
    env.close()
