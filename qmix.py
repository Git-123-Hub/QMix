from ReplayBuffer import ReplayBuffer


class QMix:
    def __init__(self, n_agents, state_dim, obs_dim, action_dim, max_episode, episode_limit):
        self.agent_net = None
        self.hyper_net = None
        self.buffer = ReplayBuffer(max_episode, n_agents, state_dim, obs_dim, action_dim, episode_limit)
