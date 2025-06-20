import matplotlib.pyplot as plt
import csv
import os

def save_rewards_to_csv(rewards, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])

def load_rewards_from_csv(filename):
    episodes, rewards = [], []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))
    return episodes, rewards

def plot_rewards(csv_path, save_path=None, window=50):
    episodes, rewards = load_rewards_from_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Reward")

    if window > 1:
        smoothed = moving_average(rewards, window)
        plt.plot(episodes[window-1:], smoothed, label=f"{window}-ep Moving Avg")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Time")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def moving_average(data, window):
    return [sum(data[i-window:i]) / window for i in range(window, len(data)+1)]
