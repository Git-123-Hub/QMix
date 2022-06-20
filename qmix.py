import numpy as np
import torch

from net import AgentNet, MixNet
from ReplayBuffer import ReplayBuffer


class QMix:
    def __init__(self, n_agents, state_dim, obs_dim, action_dim, max_episode, episode_limit, hidden_dim):
        # agent network take `obs` and `last action` as input, besides all the agents
        # use the same network, so an onehot vector is also used to distinguish different agent
        input_dim = obs_dim + n_agents + action_dim
        self.agent_net = AgentNet(n_agents, input_dim, hidden_dim, action_dim)
        self.target_agent_net = AgentNet(n_agents, input_dim, hidden_dim, action_dim)
        self.mix_net = MixNet(state_dim, 32, 3)  # todo: dimension info for mixing net
        self.target_mix_net = MixNet(state_dim, 32, 3)  # todo: mixnet_hidden_dim
        self.buffer = ReplayBuffer(max_episode, n_agents, state_dim, obs_dim, action_dim, episode_limit)
        self.n_agents = n_agents
        self.parameters = list(self.agent_net.parameters()) + list(self.mix_net.parameters())
        self.optimizer = torch.optim.RMSprop(self.parameters, lr=5e-4)
        # todo: add args for learning-rate

    def choose_action(self, obs, last_actions, avail_actions, epsilon):
        """
        choose action of an agent according to its id and observation
        """
        actions = []
        for i, (ob, last_action, avail_action) in enumerate(zip(obs, last_actions, avail_actions)):
            if np.random.uniform() < epsilon:
                avail_actions_ind = np.nonzero(avail_action)[0]
                actions.append(np.random.choice(avail_actions_ind))
                continue
            agent_onehot = np.eye(self.n_agents)[i]
            # todo: try torch.cat, write in one line
            net_input = np.hstack([ob, last_action, agent_onehot])
            net_input = torch.Tensor(net_input).unsqueeze(0)  # Torch.Size([1, input_dim])
            q_values = self.agent_net(net_input, i)
            q_values[np.array(avail_action) == 0] = -float("inf")  # unavailable actions can't be chosen
            action = torch.argmax(q_values).item()  # int
            actions.append(action)
        return actions

    def learn(self, batch_size, gamma):
        batch_size = 1  # todo: delete this
        if len(self.buffer) < batch_size:  # only start to learn when there are enough experience to sample
            # todo: try learn every step
            return

        # todo: add parameter learn_epoch, save_interval, train_interval
        obs, next_obs, state, next_state, action, action_onehot, avail_action, next_avail_action, reward, \
        terminate, mask = self.buffer.sample(batch_size)
        # there are `batch_size` episodes of experience sampled, we need to calculate q value of each step
        self.target_agent_net.init_hidden_state()  # todo: the shape of init hidden
        max_episode_length = obs.shape[1]  # obs shape: Torch.Size([episode, step, agent, obs_dim])
        # q_values = torch.zeros((batch_size, max_episode_length, self.n_agents))
        # target_q_values = torch.zeros_like(q_values)

        q_values_list = []
        target_q_values_list = []
        for step in range(max_episode_length):  # calculate q value of each step
            ob = obs[:, step]
            next_ob = next_obs[:, step]
            act = action_onehot[:, step]
            if step == 0:
                last_action = torch.zeros_like(act)
            else:
                last_action = action_onehot[:, step - 1]
            agent_onehot = torch.eye(self.n_agents).unsqueeze(0)

            net_input = torch.cat([ob, last_action, agent_onehot], dim=2)
            target_net_input = torch.cat([next_ob, act, agent_onehot], dim=2)

            q = self.agent_net(net_input)
            q = torch.gather(q, dim=2, index=action[:, 0].unsqueeze(-1)).squeeze(-1)  # Torch.Size([episode, n_agents])
            # q_values[:, step] = q.clone()
            q_values_list.append(q)

            target_q = self.target_agent_net(target_net_input)
            target_q[next_avail_action[:, step] == 0] = -float('inf')
            target_q = target_q.max(dim=2)[0]
            # target_q_values[:, step] = target_q.clone()
            target_q_values_list.append(target_q)

        q_values = torch.stack(q_values_list, dim=1)
        q_total = self.mix_net(q_values, state)

        target_q_values = torch.stack(target_q_values_list, dim=1)
        target_q_total = self.target_mix_net(target_q_values, next_state)
        target = reward + gamma * target_q_total * (1 - terminate)

        td_error = mask * (q_total - target.detach())
        loss = (td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        # todo: add args for clip_grad_norm
        torch.nn.utils.clip_grad_norm_(self.parameters, 10)
        self.optimizer.step()
