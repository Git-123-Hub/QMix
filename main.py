import argparse

import numpy as np
from matplotlib import pyplot as plt
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
    parser.add_argument('--n_episodes', type=int, default=10000, help='total episode num of training process')
    parser.add_argument('--capacity', type=int, default=5e3, help='maximum number of episode in buffer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of episode sampled each time from buffer')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='hidden dimension for rnn')
    parser.add_argument('--mix_hidden_dim', type=int, default=32, help='hidden dimension for mix net')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--clip_grad_norm', type=float, default=10, help='discount factor')
    parser.add_argument('--step_mul', type=int, default=8, help='How many game steps per agent step')
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--mini_epsilon', type=float, default=0.05)
    parser.add_argument('--epsilon_anneal_step', type=int, default=60 * 8000)

    args = parser.parse_args()
    epsilon_anneal = (args.epsilon - args.mini_epsilon) / args.epsilon_anneal_step

    env = StarCraft2Env(map_name=args.map, step_mul=args.step_mul)
    env_info = env.get_env_info()
    state_dim = env_info["state_shape"]
    obs_dim = env_info["obs_shape"]
    action_dim = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    steps_per_episode = env_info["episode_limit"]

    qmix = QMix(env_info, args.capacity, args.rnn_hidden_dim, args.mix_hidden_dim, args.lr)
    all_rewards = np.zeros(args.n_episodes)

    for cur_episode in range(args.n_episodes):
        env.reset()
        terminate = False
        win = False
        step = 0  # record the length of this episode
        episode_reward = 0
        # save the trajectory of this episode
        obs_list, state_list, action_list, avail_action_list, reward_list, terminate_list = [], [], [], [], [], []

        while not terminate:  # interact with the env for an episode
            obs = env.get_obs()
            state = env.get_state()
            # record last action of each agent for choosing action
            last_actions = np.zeros((n_agents, action_dim))
            qmix.agent_net.init_hidden_state(1)  # init hidden state at the start of the episode
            # env.render()  # Uncomment for rendering

            avail_actions = [env.get_avail_agent_actions(agent_id) for agent_id in range(n_agents)]
            actions = qmix.choose_action(obs, last_actions, avail_actions, args.epsilon)
            reward, terminate, info = env.step(actions)
            if 'battle_won' in info and info['battle_won'] is True:
                win = True
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
            if args.epsilon > args.mini_epsilon:  # anneal epsilon
                args.epsilon -= epsilon_anneal

        # episode finishes
        print(f"episode: {cur_episode + 1}, step: {step}, episode reward: {episode_reward}, epsilon: {args.epsilon}, {win}")
        all_rewards[cur_episode] = episode_reward

        # we have to save the last data and then save them all
        avail_actions = []
        for agent_id in range(n_agents):
            avail_actions.append(env.get_avail_agent_actions(agent_id))
        obs_list.append(env.get_obs())
        state_list.append(env.get_state())
        avail_action_list.append(avail_actions)
        qmix.buffer.add(obs_list, state_list, action_list, avail_action_list, reward_list, terminate_list)
        qmix.learn(args.batch_size, args.gamma, args.clip_grad_norm)

    env.close()

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.n_episodes + 1)
    ax.plot(x, all_rewards)
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of {args.map}'
    ax.set_title(title)
    plt.savefig(title)
