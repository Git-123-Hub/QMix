import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_episode, n_agents, state_dim, obs_dim, action_dim, episode_limit):
        max_episode = int(max_episode)
        self.max_episode = max_episode
        # when learning, we need to calculate q_target of each agent using `next_obs`, and we need the
        # max value of these q_targets over their avail_action, so `next_avail_action` is needed,
        # then we mix them up using `next_state` to get q_target_total.
        # In summary, we need to record the data of `obs`, `state`, `avail_action` of the `next` timestep
        # thus the length of `obs`, `state`, `avail_action` is 1 more than others
        self.obs = np.zeros([max_episode, episode_limit + 1, n_agents, obs_dim])
        self.state = np.zeros([max_episode, episode_limit + 1, state_dim])
        self.action = np.zeros([max_episode, episode_limit, n_agents, action_dim])
        self.avail_action = np.zeros([max_episode, episode_limit + 1, n_agents, action_dim])
        self.reward = np.zeros([max_episode, episode_limit])
        self.terminate = np.zeros([max_episode, episode_limit])
        self.mask = np.zeros([max_episode, episode_limit])  # if cur step an available transition

        self._index = 0
        self._size = 0

    def add(self, obs_list, state_list, action_list, avail_action_list, reward_list, terminate_list):
        """add an experience to the replay buffer"""
        episode_length = len(terminate_list)
        self.obs[self._index][:episode_length + 1] = np.array(obs_list[:])
        self.state[self._index][:episode_length + 1] = np.array(state_list[:])
        # todo: a better way of getting onehot action vector
        # action_list is a list of int, meaning the index of the action
        # we get onehot vector by setting corresponding value to 1
        for step in range(episode_length):
            for agent_num, act in enumerate(action_list[step]):
                self.action[self._index][step][agent_num][act] = 1
        self.avail_action[self._index][:episode_length + 1] = np.array(avail_action_list[:])
        self.reward[self._index][:episode_length] = np.array(reward_list)
        self.terminate[self._index][:episode_length] = np.array(terminate_list)
        self.mask[self._index][:episode_length] = 1  # indicate that these indexes are available

        self._index = (self._index + 1) % self.max_episode
        if self._size < self.max_episode:
            self._size += 1

    def sample(self, batch_size):
        """sample experience from the buffer for learning"""
        # todo: test the behavior of this method
        index = np.random.choice(range(self._size), size=batch_size, replace=False)  # without replacement
        # get the max_episode_length of these episodes to filter out useless data
        # Note that the sum of self.mask[i] indicates the length of episode i according to definition of mask
        max_episode_length = max(sum(self.mask[i]) for i in index)

        obs = torch.from_numpy(self.obs[index][:max_episode_length])
        next_obs = torch.from_numpy(self.obs[index][1:max_episode_length + 1])
        state = torch.from_numpy(self.state[index][:max_episode_length])
        next_state = torch.from_numpy(self.state[index][1:max_episode_length + 1])
        action = torch.from_numpy(self.action[index][:max_episode_length])
        avail_action = torch.from_numpy(self.avail_action[index][:max_episode_length])
        next_avail_action = torch.from_numpy(self.avail_action[index][1:max_episode_length + 1])
        reward = torch.from_numpy(self.reward[index][:max_episode_length])
        terminate = torch.from_numpy(self.terminate[index][:max_episode_length])
        mask = torch.from_numpy(self.mask[index][:max_episode_length])

        return obs, next_obs, state, next_state, action, avail_action, next_avail_action, reward, terminate, mask

    def __len__(self):
        return self._size
