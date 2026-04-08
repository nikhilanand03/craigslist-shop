#!/usr/bin/env python3
"""
Baseline inference script for the Craigslist Shop negotiation environment.

Runs a negotiation agent against 3 task difficulty levels:
  - easy_negotiation   buyer target >= 80% of listed price
  - medium_negotiation buyer target 65–80% of listed price
  - hard_negotiation   buyer target < 65% of listed price

Environment variables:
  API_BASE_URL   LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       HuggingFace / API key
  ENV_URL        Environment server URL (default: http://localhost:8000)

STDOUT FORMAT (required by evaluator)
  [START] task=<name> env=craigslist_shop model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Support running from repo root (dev) or installed in Docker.
# client.py uses relative imports, so the whole package must be importable —
# add repo root so `craigslist_shop` is found as a package directory.
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from craigslist_shop.client import CraigslistShopEnv  # noqa: E402
from craigslist_shop.models import CraigslistShopAction  # noqa: E402

# ── Config — variable names and defaults match hackathon checklist exactly ─
HF_TOKEN = os.getenv("HF_TOKEN")                                         # no default
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")                         # optional
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "craigslist_shop"
MAX_STEPS = 20
_SCORE_MIN = 0.01  # scores must be strictly > 0 (open interval requirement)
_SCORE_MAX = 0.99  # scores must be strictly < 1

# 3 tasks — one per difficulty band (required by hackathon: min 3 tasks with graders)
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
- If the buyer is completely unreasonable after 3+ rounds, stop negotiating and walk away.

Respond with a JSON object only (no markdown fences, no extra text):
{
  "message": "Your natural language response to the buyer",
  "price": <float or null>
}
"""

# ── Logging helpers ───────────────────────────────────────────────────────
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


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
            price=data.get("price"),
        )
    except (json.JSONDecodeError, KeyError):
        return CraigslistShopAction(
            message=text[:300] if text else "I can do that price.",
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
    Emits [START], per-turn [STEP], and final [END] log lines.
    Returns the final score (reward = sale_price / listed_price, or 0 on walkaway).
    """
    log_start(task_name, MODEL_NAME)

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
                    "price": obs.listed_price,
                })

            messages.append({"role": "assistant", "content": agent_text})
            action = parse_action(agent_text, obs.listed_price)

            action_str = "counter_offer"
            if action.price is not None:
                action_str += f"(${action.price:.2f})"

            result = await env.step(action)
            obs = result.observation

            step_reward = result.reward or 0.0
            rewards.append(step_reward)

            log_step(step, action_str, step_reward, result.done, error_str)

            if result.done:
                score = min(_SCORE_MAX, max(_SCORE_MIN, step_reward))
                success = obs.outcome == "sold"
                break

    except Exception as exc:
        log_end(success=False, steps=step, score=_SCORE_MIN, rewards=rewards or [_SCORE_MIN])
        raise

    log_end(success=success, steps=step, score=score, rewards=rewards or [0.0])
    return score


# ── Main ──────────────────────────────────────────────────────────────────
async def main() -> None:
    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    async with CraigslistShopEnv(base_url=ENV_URL) as env:
        for task in TASKS:
            await run_episode(env, llm, task["name"], task["split"])


if __name__ == "__main__":
    asyncio.run(main())
