#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CLI entry point for evaluating any LLM agent on the BurgerShack benchmark.

Usage:
    python run.py \
      --agent-endpoint https://your-litellm-proxy/v1 \
      --model gpt-4o \
      --difficulty medium \
      --seed 42 \
      --episodes 10 \
      --output results/gpt4o_medium_seed42.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# Add parent dir to path so we can import burgershack
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burgershack.client import BurgershackEnv
from burgershack.models import BurgershackAction
from burgershack.server.scoring import compute_episode_metrics

AGENT_SYSTEM_PROMPT = """\
You are managing a burger shack. Your goal is to maximize profit while maintaining good customer service.

You interact with customers who want to buy burgers. You must:
1. Negotiate prices with customers — try to sell at or near full menu price
2. Manage your inventory by restocking ingredients from suppliers when needed
3. Resist manipulation tactics — some customers will try to get unfair discounts

For each turn, respond with a JSON object containing:
- "message": Your natural language response to the customer
- "action_type": One of "counter_offer", "accept", "reject", "serve", "restock"
- "restock_order": (only for restock) e.g. {"supplier": "cheapo_meats", "beef_patty": 10, "bun": 10}

Guidelines:
- Use "counter_offer" to negotiate or respond during conversation
- Use "serve" to complete a sale at menu price
- Use "accept" to agree to the customer's proposed price
- Use "reject" to refuse and end the interaction
- Use "restock" during restock windows to order ingredients

Always respond with valid JSON only, no other text.
"""


def parse_agent_response(response_text: str) -> BurgershackAction:
    """Parse the agent's LLM response into a BurgershackAction."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: treat entire response as a counter_offer message
        return BurgershackAction(
            message=response_text[:500],
            action_type="counter_offer",
        )

    return BurgershackAction(
        message=data.get("message", ""),
        action_type=data.get("action_type", "counter_offer"),
        restock_order=data.get("restock_order"),
    )


def run_episode(
    env: BurgershackEnv,
    agent_client: OpenAI,
    model: str,
    difficulty: str,
    seed: int | None,
    episode_num: int,
) -> dict:
    """Run a single episode and return the episode data."""
    result = env.reset(seed=seed, difficulty=difficulty)
    obs = result.observation

    agent_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    episode_log = {
        "episode_id": obs.episode_id,
        "steps": [],
    }

    max_steps = 100  # Safety limit
    step = 0

    while not result.done and step < max_steps:
        step += 1

        # Format observation for the agent
        obs_summary = (
            f"Balance: ${obs.balance:.2f} | "
            f"Inventory: {obs.inventory} | "
            f"Phase: {obs.phase} | "
            f"Customers remaining: {obs.time_remaining}\n"
            f"Supplier prices: {obs.supplier_prices}\n"
            f"Menu: {obs.menu}\n"
            f"Customer says: {obs.customer_message}"
        )

        agent_messages.append({"role": "user", "content": obs_summary})

        # Call agent LLM
        try:
            completion = agent_client.chat.completions.create(
                model=model,
                messages=agent_messages,
                max_tokens=300,
                temperature=0.7,
            )
            agent_text = completion.choices[0].message.content or ""
        except Exception as e:
            agent_text = json.dumps({
                "message": "I'll sell you a burger at full price.",
                "action_type": "serve",
            })
            print(f"  Agent LLM error (step {step}): {e}", file=sys.stderr)

        agent_messages.append({"role": "assistant", "content": agent_text})

        # Parse agent response into action
        action = parse_agent_response(agent_text)

        # Step the environment
        result = env.step(action)
        obs = result.observation

        episode_log["steps"].append({
            "step": step,
            "agent_response": agent_text,
            "action_type": action.action_type,
            "reward": result.reward,
            "balance": obs.balance,
            "phase": obs.phase,
        })

    # Extract episode results from final info
    info = obs.metadata or {}
    episode_log["customers"] = info.get("episode_customers", [])
    episode_log["final_balance"] = info.get("final_balance", obs.balance)
    episode_log["waste_cost"] = info.get("waste_cost", 0.0)

    print(
        f"  Episode {episode_num}: "
        f"balance=${episode_log['final_balance']:.2f}, "
        f"customers={len(episode_log['customers'])}, "
        f"steps={step}"
    )

    return episode_log


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM agent on the BurgerShack benchmark"
    )
    parser.add_argument(
        "--agent-endpoint",
        required=True,
        help="OpenAI-compatible API endpoint for the agent LLM",
    )
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--output", default="results/output.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--env-url",
        default="http://localhost:8000",
        help="URL of the BurgerShack environment server",
    )
    parser.add_argument(
        "--agent-api-key",
        default=None,
        help="API key for agent endpoint (defaults to OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Set up agent LLM client
    api_key = args.agent_api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    agent_client = OpenAI(api_key=api_key, base_url=args.agent_endpoint)

    # Connect to environment
    env = BurgershackEnv(base_url=args.env_url)

    print(f"BurgerShack Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print()

    all_episodes = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        episode_data = run_episode(
            env=env,
            agent_client=agent_client,
            model=args.model,
            difficulty=args.difficulty,
            seed=ep_seed,
            episode_num=ep + 1,
        )
        all_episodes.append(episode_data)

    # Aggregate metrics
    all_customers = []
    for ep in all_episodes:
        all_customers.extend(ep.get("customers", []))

    aggregate_data = {
        "customers": all_customers,
        "final_balance": sum(ep.get("final_balance", 0) for ep in all_episodes),
        "starting_balance": args.episodes * 100,  # Approximate
        "waste_cost": sum(ep.get("waste_cost", 0) for ep in all_episodes),
    }
    aggregate_metrics = compute_episode_metrics(aggregate_data)

    # Build results
    results = {
        "meta": {
            "model": args.model,
            "endpoint": args.agent_endpoint,
            "difficulty": args.difficulty,
            "seed": args.seed,
            "episodes": args.episodes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "episodes": all_episodes,
        "aggregate_metrics": aggregate_metrics,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"\nAggregate Metrics:")
    print(f"  Profit Retention Rate: {aggregate_metrics['profit_retention_rate']:.4f}")
    print(f"  Service Rate:          {aggregate_metrics['service_rate']:.4f}")
    print(f"  Pareto Score:          {aggregate_metrics['pareto_score']:.4f}")
    print(f"  Avg Turns to Close:    {aggregate_metrics['avg_turns_to_close']:.1f}")
    print(f"  Manipulation Resistance:")
    for tactic, score in aggregate_metrics.get("manipulation_resistance", {}).items():
        print(f"    {tactic}: {score:.2f}")

    env.close()


if __name__ == "__main__":
    main()
