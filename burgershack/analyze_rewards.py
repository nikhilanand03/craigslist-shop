#!/usr/bin/env python3
"""
Analyze run results and produce a line plot of average reward
per agent strategy.

Reads all JSON files from runs/ and saves chart to analysis/.

Usage:
    python analyze_rewards.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"


def load_all_runs() -> list[dict]:
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def main():
    runs = load_all_runs()
    if not runs:
        print(f"No run files found in {RUNS_DIR}")
        return

    # Collect all episode rewards per strategy
    strategy_rewards: dict[str, list[float]] = {}
    for run in runs:
        strategy = run.get("strategy", "unknown")
        rewards = [ep.get("reward", 0.0) for ep in run.get("episodes", [])]
        strategy_rewards.setdefault(strategy, []).extend(rewards)

    # Compute stats
    strategies = sorted(strategy_rewards.keys())
    avgs = [np.mean(strategy_rewards[s]) for s in strategies]
    stds = [np.std(strategy_rewards[s]) for s in strategies]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(strategies))
    ax.plot(x, avgs, marker="o", linewidth=2, color="#2b6cb0", markersize=8, zorder=3)
    ax.fill_between(
        x,
        [a - s for a, s in zip(avgs, stds)],
        [a + s for a, s in zip(avgs, stds)],
        alpha=0.15,
        color="#2b6cb0",
    )

    # Annotate values
    for i, (avg, std) in enumerate(zip(avgs, stds)):
        ax.annotate(
            f"{avg:.3f}",
            (i, avg),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.set_xlabel("Agent Strategy", fontsize=12)
    ax.set_title("Average Reward by Agent Strategy", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    ANALYSIS_DIR.mkdir(exist_ok=True)
    output_path = ANALYSIS_DIR / "avg_reward_by_strategy.png"
    fig.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")

    # Print summary
    print(f"\n{'Strategy':<25} {'Avg Reward':>12} {'Std':>8} {'Episodes':>10}")
    print("-" * 57)
    for s, avg, std in zip(strategies, avgs, stds):
        print(f"{s:<25} {avg:>12.4f} {std:>8.4f} {len(strategy_rewards[s]):>10}")


if __name__ == "__main__":
    main()
