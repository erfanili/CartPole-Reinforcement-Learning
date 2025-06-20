import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trainer.train_q_learning import train_q_learning
from utils.plot import save_rewards_to_csv, plot_rewards


if __name__ == "__main__":
    config = {
        'alpha': 0.01,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.9999,
        'epsilon_min': 0.01,
        'n_bins': (500,),
        'obs_low': [-0.2],
        'obs_high': [0.2],
    }

    rewards = train_q_learning(config, episodes=50000)
    save_rewards_to_csv(rewards, "data/q_learning_rewards.csv")
    plot_rewards("data/q_learning_rewards.csv", save_path="data/reward_plot.png")
    
