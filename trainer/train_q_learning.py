from env.cartpole import CartPoleEnv
from agent.q_learning_agent import QLearningAgent

import numpy as np
import csv
import os

def train_q_learning(config, episodes=5):
    env = CartPoleEnv()
    agent = QLearningAgent(env, config)

    rewards = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes} | Reward: {total_reward:.2f} | Steps: {env.steps:.3f}")
    agent.save("data/q_table.npy")
    return rewards

