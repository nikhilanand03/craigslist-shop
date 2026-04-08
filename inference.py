"""
Inference Script — Craigslist Shop Negotiation Benchmark
===================================

Environment variables (set locally or as HF Space secrets):
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name (optional, for from_docker_image())

Stdout format follows the OpenEnv competition spec:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from craigslist_shop import CraigslistShopAction, CraigslistShopEnv

HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://nikhilanand-craigslist-shop.hf.space")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
BENCHMARK = "craigslist_shop"
MAX_STEPS = 20
TEMPERATURE = 0.7
MAX_TOKENS = 300

# Run 5 tasks from the test set
TASK_INDICES = [0, 1, 2, 3, 4]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an experienced seller on Craigslist. Your goal is to sell items at
    the highest price possible — as close to the listed price as you can.

    You negotiate with buyers who want a lower price. Your strategy:
    1. Start by holding firm at or near the listed price
    2. Emphasize the item's value, condition, and unique features
    3. Make small concessions (2-5% at a time) only if the buyer pushes back
    4. Never go below 85% of the listed price
    5. Use persuasion: mention other interested buyers, highlight quality, create urgency
    6. Be conversational, confident, and friendly

    You must respond with a JSON object (no markdown, no extra text):
    {
      "message": "Your natural language response to the buyer",
      "price": 100.00
    }

    - "message": What you say to the buyer
    - "price": The price you are currently offering (always include this)

    The buyer decides whether to accept, counter, or walk away.
    Always respond with valid JSON only, no other text.
""")


# ── Logging helpers ──────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent logic ──────────────────────────────────────────────────────────

def build_user_prompt(obs) -> str:
    parts = [
        f"Item: {obs.item_title} ({obs.item_category})",
        f"Description: {obs.item_description[:300]}",
        f"Listed price: ${obs.listed_price:.2f}",
        f"Your last offered price: {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}",
        f"Turn: {obs.turn}",
    ]
    if obs.system_message:
        parts.append(f"\nSystem: {obs.system_message}")
    if obs.customer_message:
        parts.append(f"\nBuyer says: {obs.customer_message}")
    return "\n".join(parts)


def parse_agent_response(text: str) -> CraigslistShopAction:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return CraigslistShopAction(message=text[:500])

    return CraigslistShopAction(
        message=data.get("message") or "",
        price=data.get("price"),
    )


def get_agent_response(client: OpenAI, messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"message": "What is your offer?", "price": null}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"message": "What is your offer?", "price": null}'


# ── Run one task ─────────────────────────────────────────────────────────

async def run_task(env: CraigslistShopEnv, client: OpenAI, task_index: int) -> float:
    """Run a single negotiation task. Returns the score in [0, 1]."""
    task_name = f"negotiation_{task_index}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_index=task_index, split="test")
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Build prompt from observation
            user_prompt = build_user_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            # Get agent response
            agent_text = get_agent_response(client, messages)
            messages.append({"role": "assistant", "content": agent_text})

            # Parse and step
            action = parse_agent_response(agent_text)
            action_str = f"offer(msg='{action.message[:50]}',price={action.price})"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.system_message if "ERROR" in obs.system_message else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Score is the final reward (sale_price / listed_price), already in [0, 1]
        # For walkaway it's 0.0, for a sale it's the price retention ratio
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0  # any sale is a success

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ─────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] Using model: {MODEL_NAME}", flush=True)

    env = CraigslistShopEnv(base_url=HF_SPACE_URL)
    await env.connect()

    try:
        scores = []
        for task_idx in TASK_INDICES:
            score = await run_task(env, client, task_idx)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"\n[SUMMARY] tasks={len(scores)} avg_score={avg_score:.3f} scores={','.join(f'{s:.3f}' for s in scores)}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
