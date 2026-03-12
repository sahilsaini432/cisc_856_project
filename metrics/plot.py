import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_progress(total_rewards, alg_name, window=10):
    n = len(total_rewards)
    episodes = np.arange(1, n + 1)
    rolling_avg = np.convolve(total_rewards, np.ones(window) / window, mode="valid")

    success_rate = sum(r > 0 for r in total_rewards) / n * 100
    avg_reward = np.mean(total_rewards)
    min_reward = np.min(total_rewards)
    max_reward = np.max(total_rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, total_rewards, alpha=0.3, label="Episode reward")
    ax.plot(episodes[window - 1 :], rolling_avg, label=f"{window}-episode avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{alg_name} — MCTS Performance")
    ax.legend(loc="upper left")

    stats = (
        f"Episodes: {n}\n"
        f"Success rate: {success_rate:.1f}%\n"
        f"Avg reward: {avg_reward:.4f}\n"
        f"Min / Max: {min_reward:.4f} / {max_reward:.4f}"
    )
    ax.text(
        0.98,
        0.05,
        stats,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    filename = f"/graphs/{alg_name}_progress.png"
    plt.savefig(filename)
    plt.show()
