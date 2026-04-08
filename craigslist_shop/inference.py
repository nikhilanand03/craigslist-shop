#!/usr/bin/env python3
"""
Baseline inference script for the Craigslist Shop negotiation environment.

Runs a negotiation agent against 3 task difficulty levels:
  - easy_negotiation   buyer target >= 80% of listed price
  - medium_negotiation buyer target 65-80% of listed price
  - hard_negotiation   buyer target < 65% of listed price

Environment variables:
  API_BASE_URL   LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HuggingFace / API key
  ENV_URL        Environment server URL (default: http://localhost:8000)

STDOUT FORMAT
  [START] task=<name> env=craigslist_shop model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

# In Docker the package is installed via uv sync (no-op).
# In development, add the repo root so `craigslist_shop` is findable as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from craigslist_shop.client import CraigslistShopEnv  # noqa: E402
from craigslist_shop.models import CraigslistShopAction  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "craigslist_shop"
MAX_STEPS = 20

# 3 tasks — one per difficulty band (satisfies hackathon min-3-tasks requirement)
TASKS = [
    {"name": "easy_negotiation",   "split": "easy"},
    {"name": "medium_negotiation", "split": "medium"},
    {"name": "hard_negotiation",   "split": "hard"},
]

# ── Agent system prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a skilled Craigslist seller. Sell the item as close to the listed price
as possible while keeping the buyer engaged.

Guidelines:
- Start at or near the listed price. Justify the price with item details.
- Make small concessions (3-5% at a time) only if the buyer pushes back.
- Never go below 80% of the listed price.
- If the buyer's offer is within 10% of listed price, accept immediately.
- Use "reject" only if the buyer is completely unreasonable after 3+ rounds.

Respond with a JSON object only (no markdown fences, no extra text):
{
  "message": "Your natural language response to the buyer",
  "action_type": "counter_offer | accept | reject",
  "price": <float or null>
}
"""

# ── Logging helpers (exact spec format) ──────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.2f} rewards={rewards_str}", flush=True)


# ── Action parsing ────────────────────────────────────────────────────────
def parse_action(text: str, listed_price: float) -> CraigslistShopAction:
    """Parse LLM JSON output into a CraigslistShopAction. Falls back gracefully."""
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
            break
    try:
        data = json.loads(text)
        return CraigslistShopAction(
            message=data.get("message", ""),
            action_type=data.get("action_type", "counter_offer"),
            price=data.get("price"),
        )
    except (json.JSONDecodeError, KeyError):
        return CraigslistShopAction(
            message=text[:300] if text else "I can do that price.",
            action_type="counter_offer",
            price=listed_price,
        )


# ── Episode runner ────────────────────────────────────────────────────────
async def run_episode(
    env: CraigslistShopEnv,
    llm: OpenAI,
    task_name: str,
    split: str,
) -> float:
    """
    Run one negotiation episode.
    Emits [START], one [STEP] per turn, and [END] — always, even on exception.
    Returns the final score clamped to [0, 1].
    """
    log_start(task_name, BENCHMARK, MODEL_NAME)

    result = await env.reset(task_index=0, split=split)
    obs = result.observation

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    step = 0
    rewards: list = []
    score = 0.0
    success = False

    try:
        while not result.done and step < MAX_STEPS:
            step += 1

            obs_lines = [
                f"Item: {obs.item_title} ({obs.item_category})",
                f"Description: {obs.item_description[:300]}",
                f"Listed price: ${obs.listed_price:.2f}",
                f"Your last offered price: {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}",
                f"Turn: {obs.turn}",
            ]
            if obs.customer_message:
                obs_lines.append(f"Buyer says: {obs.customer_message}")
            if obs.system_message:
                obs_lines.append(f"System: {obs.system_message}")
            messages.append({"role": "user", "content": "\n".join(obs_lines)})

            error_str = None
            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                )
                agent_text = completion.choices[0].message.content or ""
            except Exception as e:
                error_str = str(e)[:120]
                agent_text = json.dumps({
                    "message": f"The price is ${obs.listed_price:.2f}.",
                    "action_type": "counter_offer",
                    "price": obs.listed_price,
                })

            messages.append({"role": "assistant", "content": agent_text})
            action = parse_action(agent_text, obs.listed_price)

            action_str = action.action_type
            if action.price is not None:
                action_str += f"(${action.price:.2f})"

            result = await env.step(action)
            obs = result.observation

            step_reward = result.reward or 0.0
            rewards.append(step_reward)

            log_step(step, action_str, step_reward, result.done, error_str)

            if result.done:
                # Clamp score to [0, 1] as required by hackathon spec
                score = min(1.0, max(0.0, step_reward))
                success = obs.outcome == "sold"
                break

    except Exception:
        log_end(success=False, steps=step, score=0.0, rewards=rewards or [0.0])
        raise

    log_end(success=success, steps=step, score=score, rewards=rewards or [0.0])
    return score


# ── Main ──────────────────────────────────────────────────────────────────
async def main() -> None:
    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    async with CraigslistShopEnv(base_url=ENV_URL) as env:
        for task in TASKS:
            await run_episode(env, llm, task["name"], task["split"])


if __name__ == "__main__":
    asyncio.run(main())
