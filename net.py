from abc import ABCMeta, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module


class Net(Module, metaclass=ABCMeta):
    def __init__(self):
        super(Net, self).__init__()

    @abstractmethod
    def forward(self, *args):
        pass

    def update(self, other, tau):
        """
        update the parameter of `self` using `other` with a proportion of `tau`
        """
        for self_p, other_p in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(tau * other_p.data + (1.0 - tau) * self_p.data)


class AgentNet(Net):
    def __init__(self, n_agents, input_dim, hidden_dim, output_dim):
        super(AgentNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.hidden_state = None
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def init_hidden_state(self, episode_num):
        # todo: check data dimension
        # hidden state shape[1, hidden_dim] for each agent
        self.hidden_state = torch.zeros((self.n_agents, episode_num, self.hidden_dim))

    def _get_q_value(self, obs, agent_i):
        """
        calculate q value of `agent_i` given its local `obs`, return a tensor with length `action_dim`,
        meaning the q value of each action
        """
        x = F.relu(self.fc1(obs))
        hidden = self.hidden_state[agent_i].clone()
        hidden = self.rnn(x, hidden)
        q = self.fc2(hidden)
        self.hidden_state[agent_i] = hidden
        return q

    def forward(self, obs, agent_i=None):
        # todo: refactor this method, choose action also don't need agent_i
        if agent_i is None:
            # no agent_i specified, return q values of all agents
            q_values = [self._get_q_value(obs[:, i], i) for i in range(self.n_agents)]
            return torch.stack(q_values, dim=1)  # Torch.size([episode, n_agents, action_dim])
        else:
            return self._get_q_value(obs, agent_i).squeeze(0)  # Torch.size([action_dim])


class MixNet(Net):
    def __init__(self, in_dim, hidden_dim, n_agents):
        super(MixNet, self).__init__()
        # the structure of network is as described in paper
        self.w1 = nn.Linear(in_dim, n_agents * hidden_dim)
        self.b1 = nn.Linear(in_dim, hidden_dim)

        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.b2 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1))
        self.n_agents = n_agents
        self.in_dim = in_dim  # state_dim
        self.hidden_dim = hidden_dim

    def forward(self, q_values, state):
        """
        get q_total by mixing `q_values` with weight and bias generated from `state`
        """
        episode_num = q_values.shape[0]
        episode_length = q_values.shape[1]
        q_values = q_values.view(-1, 1, self.n_agents)
        state = state.reshape(-1, self.in_dim)

        w1 = torch.abs(self.w1(state)).view(-1, self.n_agents, self.hidden_dim)
        b1 = self.b1(state).view(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.w2(state)).view(-1, self.hidden_dim, 1)
        b2 = self.b2(state).view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(episode_num, episode_length)  # q_total of each episode of each step
