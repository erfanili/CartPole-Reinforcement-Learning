import numpy as np

class Discretizer:
    def __init__(self, config):
        self.n_bins = config.get('n_bins', (30))  # (x, x_dot, theta, theta_dot)
        self.low = np.array(config.get('obs_low', [-0.6]))
        self.high = np.array(config.get('obs_high', [0.6]))

        self.bin_width = (self.high - self.low) / self.n_bins

    def discretize(self, obs):
        clipped = np.clip(obs, self.low, self.high)
        ratios = (clipped - self.low) / self.bin_width
        indices = np.floor(ratios).astype(int)
        indices = np.clip(indices, 0, np.array(self.n_bins) - 1)
        return tuple(indices)
