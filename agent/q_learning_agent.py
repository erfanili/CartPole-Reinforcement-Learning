import numpy as np
from agent.base_agent import BaseAgent
from utils.discretizer import Discretizer

class QLearningAgent(BaseAgent):
    def __init__(self, env, config):
        self.action_space = [-1,1]
        self.action_to_index = {a: i for i, a in enumerate(self.action_space)}
        self.index_to_action = {i: a for i, a in enumerate(self.action_space)}

        self.obs_space = env.reset().shape[0]
        self.discretizer = Discretizer(config)

        self.state_bins = self.discretizer.n_bins
        self.q_table = np.zeros(self.state_bins + (len(self.action_space),))

        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.999)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)

    def get_action(self, obs):
        state = self.discretizer.discretize(obs)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        action_idx = np.argmax(self.q_table[state])
        return self.index_to_action[action_idx]


    def update(self, obs, action, reward, next_obs, done):
        state = self.discretizer.discretize(obs)
        next_state = self.discretizer.discretize(next_obs)

        action_idx = self.action_to_index[action]
        best_next_action_idx = np.argmax(self.q_table[next_state])

        td_target = reward + self.gamma * self.q_table[next_state][best_next_action_idx] * (not done)
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.alpha * td_error

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

