import numpy as np

class CartPoleEnv:
    def __init__(self):
        self.x_limit = 1
        self.theta_limit = 1
        self.max_steps = 200
        self.reset()

    def reset(self):
        # self.omega = np.random.uniform(-0.05, 0.05)
        self.pole_angle = np.random.uniform(-0.2, 0.2)
        # self.cart_vel = np.random.uniform(-0.05, 0.05)
        self.cart_pos = np.random.uniform(-0.05, 0.05)

        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.pole_angle], dtype=np.float32)

    def step(self, action):
        """ action: 0 = left, 1 = right """
        force = action
        current = self.pole_angle
        self.pole_angle += 0.1*self.pole_angle - 0.02*force


        self.steps += 1

        done = (
            abs(self.pole_angle) > self.theta_limit or
            self.steps >= self.max_steps
        )
        reward = max(0.0, 0.3 - abs(self.pole_angle)) if not done else 0.0
        return self._get_obs(), reward, done, {}

    def render(self):
        print(f"Cart Pos: {self.cart_pos:.2f}, Pole Angle: {self.pole_angle:.2f}")

