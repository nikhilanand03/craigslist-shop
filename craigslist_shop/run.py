#!/usr/bin/env python3
"""
CLI entry point for evaluating any LLM agent on the Craigslist Shop negotiation benchmark.

Usage:
    python run.py \
      --agent-endpoint https://your-litellm-proxy/v1 \
      --model gpt-4o \
      --seed 42 \
      --episodes 10 \
      --output results/gpt4o_seed42.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# Add parent dir to path so we can import the environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from craigslist_shop.client import CraigslistShopEnv
from craigslist_shop.models import CraigslistShopAction

AGENT_SYSTEM_PROMPT = """\
You are a seller on Craigslist. Your goal is to sell items at the highest \
price possible — as close to (or above) the listed price as you can.

You interact with buyers who want to negotiate. You must:
1. Negotiate prices — try to sell at or near the listed price
2. Resist manipulation tactics — some buyers will try to get unfair discounts
3. Use the item description to justify your price

For each turn, respond with a JSON object containing:
- "message": Your natural language response to the buyer
- "price": The price you are currently offering

The buyer will decide whether to accept your price, counter, or walk away.
You cannot force a sale — you can only negotiate.

Always respond with valid JSON only, no other text.
"""


def parse_agent_response(response_text: str) -> CraigslistShopAction:
    """Parse the agent's LLM response into a CraigslistShopAction."""
    text = response_text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return CraigslistShopAction(message=response_text[:500])

    return CraigslistShopAction(
        message=data.get("message", ""),
        price=data.get("price"),
    )


def run_episode(
    env: CraigslistShopEnv,
    agent_client: OpenAI,
    model: str,
    seed: int | None,
    episode_num: int,
) -> dict:
    """Run a single episode and return the episode data."""
    result = env.reset(seed=seed)
    obs = result.observation

    agent_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    episode_log = {
        "episode_id": obs.episode_id,
        "item_title": obs.item_title,
        "item_category": obs.item_category,
        "listed_price": obs.listed_price,
        "steps": [],
    }

    max_steps = 30
    step = 0

    while not result.done and step < max_steps:
        step += 1

        obs_summary = (
            f"Item: {obs.item_title} ({obs.item_category})\n"
            f"Description: {obs.item_description[:300]}\n"
            f"Listed price: ${obs.listed_price:.2f}\n"
            f"Your last offered price: "
            f"{'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}\n"
            f"Turn: {obs.turn}\n"
            f"Buyer says: {obs.customer_message}"
        )

        agent_messages.append({"role": "user", "content": obs_summary})

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
                "message": f"The price is ${obs.listed_price:.2f}.",
                "price": obs.listed_price,
            })
            print(f"  Agent LLM error (step {step}): {e}", file=sys.stderr)

        agent_messages.append({"role": "assistant", "content": agent_text})

        action = parse_agent_response(agent_text)

        result = env.step(action)
        obs = result.observation

        episode_log["steps"].append({
            "step": step,
            "agent_response": agent_text,
            "action_price": action.price,
            "customer_message": obs.customer_message,
            "reward": result.reward,
        })

    info = obs.metadata or {}
    episode_log["outcome"] = info.get("outcome", obs.outcome)
    episode_log["sale_price"] = info.get("sale_price", obs.sale_price)
    episode_log["reward"] = info.get("reward", result.reward or 0)
    episode_log["turns"] = obs.turn

    print(
        f"  Episode {episode_num}: "
        f"{obs.item_title[:30]} | "
        f"outcome={episode_log['outcome']}, "
        f"sale=${episode_log['sale_price']:.2f}, "
        f"listed=${episode_log['listed_price']:.2f}, "
        f"reward={episode_log['reward']:.4f}"
    )

    return episode_log


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM agent on the Craigslist Shop negotiation benchmark"
    )
    parser.add_argument(
        "--agent-endpoint",
        required=True,
        help="OpenAI-compatible API endpoint for the agent LLM",
    )
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--output", default="results/output.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--env-url",
        default="http://localhost:8000",
        help="URL of the Craigslist Shop environment server",
    )
    parser.add_argument(
        "--agent-api-key",
        default=None,
        help="API key for agent endpoint (defaults to OPENAI_API_KEY env var)",
    )

    args = parser.parse_args()

    api_key = args.agent_api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    agent_client = OpenAI(api_key=api_key, base_url=args.agent_endpoint)

    env = CraigslistShopEnv(base_url=args.env_url)

    print(f"Craigslist Shop Negotiation Benchmark")
    print(f"  Model: {args.model}")
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
            seed=ep_seed,
            episode_num=ep + 1,
        )
        all_episodes.append(episode_data)

    # Aggregate
    rewards = [e["reward"] for e in all_episodes]
    sold = [e for e in all_episodes if e["outcome"] == "sold"]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0

    results = {
        "meta": {
            "model": args.model,
            "endpoint": args.agent_endpoint,
            "seed": args.seed,
            "episodes": args.episodes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "episodes": all_episodes,
        "aggregate": {
            "avg_reward": avg_reward,
            "sale_rate": len(sold) / len(all_episodes) if all_episodes else 0,
            "total_episodes": len(all_episodes),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"  Avg reward: {avg_reward:.4f}")
    print(f"  Sale rate:  {len(sold)}/{len(all_episodes)}")

    env.close()


if __name__ == "__main__":
    main()
