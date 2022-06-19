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

    def init_hidden_state(self):
        # todo: check data dimension
        self.hidden_state = torch.zeros((self.n_agents, self.hidden_dim))

    def forward(self, state, agent_i):
        x = F.relu(self.fc1(state))
        # self.hidden_state = self.hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # todo: change dimension for hidden state
        self.hidden_state[agent_i] = self.rnn(x, self.hidden_state[agent_i])
        q = self.fc2(self.hidden_state)
        return q, self.hidden_state


class MixNet(Net):
    def __init__(self, in_dim, hidden_dim):
        super(MixNet, self).__init__()

        # todo: data dimension
        # the structure of network is as described in paper
        self.w1 = nn.Linear(in_dim, args.n_agents * hidden_dim)
        self.b1 = nn.Linear(in_dim, hidden_dim)

        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.b2 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1))

    def forward(self, q_values):
        """
        get q_total by mixing q_values with weight and bias generated
        """
        pass
