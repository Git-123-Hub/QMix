import numpy as np
import torch

from net import AgentNet, MixNet
from ReplayBuffer import ReplayBuffer


class QMix:
    def __init__(self, n_agents, state_dim, obs_dim, action_dim, max_episode, episode_limit, hidden_dim):
        # agent network take `obs` and `last action` as input, besides all the agents
        # use the same network, so an onehot vector is also used to distinguish different agent
        input_dim = obs_dim + n_agents + action_dim
        self.agent_net = AgentNet(3, input_dim, hidden_dim, action_dim)  # output q value for each action
        self.mix_net = MixNet()  # todo: dimension info for mixing net
        self.buffer = ReplayBuffer(max_episode, n_agents, state_dim, obs_dim, action_dim, episode_limit)
        self.n_agents = n_agents

    def choose_action(self, obs, last_actions, avail_actions, epsilon):
        """
        choose action of an agent according to its id and observation
        """
        actions = []
        for i, ob, last_action, avail_action in enumerate(zip(obs, last_actions, avail_actions)):
            if np.random.uniform() < epsilon:
                avail_actions_ind = np.nonzero(avail_action)[0]
                actions.append(np.random.choice(avail_actions_ind))
                continue
            agent_onehot = np.eye(self.n_agents)[i]
            # todo: 观察数据维度，try torch.cat, write in one line
            net_input = np.stack(obs, last_action, agent_onehot)
            net_input = torch.from_numpy(net_input)
            q_values = self.agent_net(net_input, i)
            q_values[avail_action == 0.0] = - float("inf")  # unavailable actions can't be chosen
            action = torch.argmax(q_values)
            actions.append(action)
        return actions

    def learn(self, batch_size):
        if len(self.buffer) < batch_size:  # only start to learn when there are enough experience to sample
            # todo: try learn every step
            return

        # todo: add parameter learn_epoch, save_interval, train_interval
        obs, next_obs, state, next_state, action, avail_action, next_avail_action, \
        reward, terminate, mask = self.buffer.sample(batch_size)
