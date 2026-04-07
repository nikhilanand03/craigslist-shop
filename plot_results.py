"""
Generate analysis plots from strategy evaluation runs.

Usage:
    python plot_results.py                   # reads from runs_analysis/
    python plot_results.py --suffix exp_v2   # reads from runs_exp_v2/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot strategy evaluation results")
    parser.add_argument("--suffix", default="analysis",
                        help="Suffix of the runs directory to read from (default: analysis)")
    args = parser.parse_args()

    runs_dir = Path(f"runs_{args.suffix}")
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        sys.exit(1)

    # Load all run files
    results = {}
    for f in sorted(runs_dir.glob("*.json")):
        data = json.loads(f.read_text())
        strategy = data["strategy"]
        results[strategy] = {
            "avg_reward": data["aggregate"]["avg_reward"],
            "std_reward": data["aggregate"]["std_reward"],
            "service_rate": data["aggregate"]["service_rate"],
            "avg_price_retention": data["aggregate"]["avg_price_retention"],
            "episodes": data["episodes"],
        }

    if not results:
        print("No results found")
        sys.exit(1)

    # Sort by avg reward
    strategies = sorted(results.keys(), key=lambda s: results[s]["avg_reward"])
    avg_rewards = [results[s]["avg_reward"] for s in strategies]
    std_rewards = [results[s]["std_reward"] for s in strategies]
    service_rates = [results[s]["service_rate"] for s in strategies]
    retentions = [results[s]["avg_price_retention"] for s in strategies]

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(strategies))

    # Avg reward with std error bars
    ax = axes[0]
    ax.bar(x, avg_rewards, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, avg_rewards, yerr=std_rewards, fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    for i, (v, s) in enumerate(zip(avg_rewards, std_rewards)):
        ax.text(i, v + s + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward by Strategy")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Sale rate
    ax = axes[1]
    ax.bar(x, [r * 100 for r in service_rates], color="#55A868", alpha=0.8, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(service_rates):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel("Sale Rate (%)")
    ax.set_title("Sale Rate by Strategy")
    ax.set_ylim(0, 110)

    # Price retention
    ax = axes[2]
    ax.bar(x, [r * 100 for r in retentions], color="#C44E52", alpha=0.8, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(retentions):
        ax.text(i, v * 100 + 1, f"{v:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel("Price Retention (%)")
    ax.set_title("Avg Price Retention by Strategy (sold only)")
    ax.set_ylim(0, 110)

    plt.suptitle("Craigslist Shop — Strategy Evaluation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = analysis_dir / f"strategy_evaluation_{args.suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # Print summary table
    print(f"\n{'Strategy':<20} {'Avg Reward':>10} {'Std':>8} {'Sale Rate':>10} {'Price Ret':>10}")
    print("-" * 62)
    for s in strategies:
        r = results[s]
        print(f"{s:<20} {r['avg_reward']:>10.4f} {r['std_reward']:>8.4f} {r['service_rate']:>9.0%} {r['avg_price_retention']:>9.1%}")


if __name__ == "__main__":
    main()
