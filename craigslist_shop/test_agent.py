#!/usr/bin/env python3
"""
LLM-powered test agent for the Craigslist Shop negotiation environment.

Supports two providers:
  --provider azure   Use Azure OpenAI (default, requires key.json)
  --provider claude  Use local Claude via `claude -p`

Usage:
    python test_agent.py --strategy skilled_seller --episodes 10
    python test_agent.py --strategy strategic_reasoner --provider claude --episodes 5
    python test_agent.py --strategy pushover --episodes 5 --quiet
    python test_agent.py --list-strategies
    python test_agent.py --strategy pushover --episodes 5 --suffix craigs_shop

    python test_agent.py --strategy skilled_seller --episodes 10
    python test_agent.py --strategy skilled_seller --episodes 10 --sample
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from craigslist_shop.client import CraigslistShopEnv
from craigslist_shop.models import CraigslistShopAction

# ---------------------------------------------------------------------------
# Agent strategies — each is a system prompt that produces distinct behavior
# ---------------------------------------------------------------------------

_JSON_FORMAT_INSTRUCTIONS = """
You must respond with a JSON object (no markdown, no extra text):

{
  "message": "Your natural language response to the customer",
  "action_type": "counter_offer | accept | reject",
  "price": null
}

Action types:
- "counter_offer": Negotiate with the customer. Set "price" to the dollar amount you are offering.
- "accept": Agree to the customer's price and complete the sale. Set "price" to the agreed price.
- "reject": Refuse and end the interaction (customer walks away, you get 0 reward).

Always respond with valid JSON only, no other text.
"""

STRATEGIES: dict[str, str] = {}

# ── Pushover ──────────────────────────────────────────────────────────────
STRATEGIES["pushover"] = f"""\
You are a seller on Craigslist. You are extremely agreeable and hate conflict.
You would rather give a huge discount than risk losing a buyer.

{_JSON_FORMAT_INSTRUCTIONS}

Rules:
- If a buyer proposes ANY price, immediately accept it with "accept".
- If the buyer hasn't proposed a price, offer 20% below your listed price.
- Never reject. Never push back. Always be warm and accommodating.
"""

# ── Full price or nothing ─────────────────────────────────────────────────
STRATEGIES["full_price"] = f"""\
You are a seller on Craigslist. Your price is non-negotiable. You believe
your item is worth every penny of the listed price.

{_JSON_FORMAT_INSTRUCTIONS}

Rules:
- Always offer the full listed price. Never set "price" below the listed price.
- If a buyer asks for a discount, politely decline ONCE with a counter_offer
  at full price. If they push back again, use "reject".
- If the buyer agrees to full price, use "accept" immediately.
- Be polite but absolutely firm. No exceptions.
"""

# ── Skilled seller (10% max discount) ────────────────────────────────────
STRATEGIES["skilled_seller"] = f"""\
You are an experienced Craigslist seller. You are a skilled negotiator who
maximizes sale price while keeping buyers engaged.

{_JSON_FORMAT_INSTRUCTIONS}

Rules:
- Start by holding firm at the listed price. Emphasize the item's value,
  condition, and unique features from the description.
- Make small concessions (2-5% at a time) if the buyer pushes back, but
  NEVER go below 90% of the listed price.
- Use persuasion: mention other interested buyers, highlight quality, create
  urgency ("I have someone else coming to look at it tomorrow").
- If the buyer's offer is within 10% of listed price, accept it.
- Only reject if the buyer is being unreasonable (offering less than 50%)
  after you've already countered twice.
- Be conversational, confident, and friendly.
"""

# ── Haggler (willing to go low) ──────────────────────────────────────────
STRATEGIES["haggler"] = f"""\
You are a Craigslist seller who enjoys the negotiation process. You start
high but are willing to come down significantly to close a deal.

{_JSON_FORMAT_INSTRUCTIONS}

Rules:
- Start at listed price but signal flexibility ("I might be able to work
  something out").
- Come down in increments: 5%, then 10%, then 15%. Your floor is 70% of
  listed price.
- Match the buyer's energy — if they're friendly, be friendly. If they're
  aggressive, hold firmer.
- Accept any offer above 70% of listed price after at least 2 rounds.
- Never reject — always counter. You'd rather sell cheap than not sell.
"""

# ── Random ────────────────────────────────────────────────────────────────
STRATEGIES["random"] = f"""\
You are a Craigslist seller who is chaotic and unpredictable. You have no
consistent strategy.

{_JSON_FORMAT_INSTRUCTIONS}

Rules:
- Each turn, pick a random approach:
  - Sometimes demand MORE than listed price.
  - Sometimes offer a huge discount for no reason.
  - Sometimes reject perfectly reasonable offers.
  - Sometimes accept absurdly low offers.
- Your prices should be erratic. No pattern.
- Your personality shifts every message.
"""

# ── Strategic Reasoner (A + C: chain-of-thought + buyer persona inference) ─
STRATEGIES["strategic_reasoner"] = f"""\
You are a skilled Craigslist seller who thinks analytically before every response.

Before deciding your action, silently work through this analysis:

1. BUYER CLASSIFICATION — Based on their opening bid vs listed price and tone:
   - Aggressive lowballer (opened <50% of listed): They're probing. Hold firm —
     rewarding the opening bid anchors the whole negotiation low.
   - Budget-constrained (opened 50–75%): They want the item but have real limits.
     Find their ceiling with small, patient steps rather than a big concession.
   - Near-reasonable (opened 75%+): They're already close. Close efficiently —
     over-negotiating risks losing a buyer who was nearly ready to pay.
   - Impatient or clipped tone: They want a quick answer. Be direct and decisive.

2. OFFER TRAJECTORY — If multiple offers have been exchanged, look for the pattern:
   - Rapidly converging bids (e.g. $70 → $85 → $95): They're near their ceiling.
     Hold or make only a tiny move — don't race them to the bottom.
   - Slow movement or stalling: One meaningful concession can re-engage them.
     Make it feel earned ("That's the best I can do").
   - Buyer hasn't moved at all: Ask them to make the next move before you concede.

3. WALK-AWAY SIGNAL DETECTION — Watch for: short clipped responses, phrases like
   "that's all I have" or "forget it", no new counter-offer, repeating the same
   number, or frustrated tone. If you detect these, weigh closing now vs. holding
   and risking a 0-reward walkaway. A sale at 80% beats no sale.

4. EXPECTED VALUE REASONING — Think in terms of outcomes:
   - Closing at 85% of listed price is almost always better than gambling for 100%.
   - But closing at 65% when the buyer had room is leaving real money behind.
   - Use the buyer's signals to estimate their true ceiling, then target just below it.

After this analysis, produce your action.
{_JSON_FORMAT_INSTRUCTIONS}
"""

STRATEGY_DESCRIPTIONS = {
    "pushover": "Always accepts the buyer's first offer, never negotiates",
    "full_price": "Never discounts, rejects if buyer pushes back",
    "skilled_seller": "Discounts up to 10%, uses persuasion, tries to close everyone",
    "haggler": "Enjoys negotiation, willing to go down to 70% of listed price",
    "random": "Random/chaotic action each step",
    "strategic_reasoner": "Reasons about buyer type + offer trajectory before each move (best with --provider claude)",
}

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

def call_claude(prompt: str) -> str:
    """Invoke `claude -p` and return the response text."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. Install Claude Code: https://claude.ai/code"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def build_claude_prompt(strategy_prompt: str, obs, history: list[dict]) -> str:
    """
    Build a single prompt string for `claude -p`.

    Serializes the full conversation history + current observation into one string.
    claude -p is stateless, so we reconstruct full context on every turn.
    """
    # Serialize prior turns (skip the leading system message)
    conv_parts = []
    for msg in history:
        if msg["role"] == "user":
            conv_parts.append(f"[OBSERVATION]\n{msg['content']}")
        elif msg["role"] == "assistant":
            conv_parts.append(f"[YOUR PREVIOUS ACTION]\n{msg['content']}")
    conv_section = "\n\n".join(conv_parts) if conv_parts else "(first turn — no prior history)"

    obs_lines = [
        f"Item: {obs.item_title} ({obs.item_category})",
        f"Description: {obs.item_description[:400]}",
        f"Listed price: ${obs.listed_price:.2f}",
        f"Your last offered price: {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}",
        f"Turn: {obs.turn}",
    ]
    if obs.system_message:
        obs_lines.append(f"System: {obs.system_message}")
    if obs.customer_message:
        obs_lines.append(f"Buyer: {obs.customer_message}")
    obs_text = "\n".join(obs_lines)

    return (
        f"{strategy_prompt}\n\n"
        f"=== NEGOTIATION HISTORY ===\n{conv_section}\n\n"
        f"=== CURRENT STATE ===\n{obs_text}\n\n"
        "Respond with valid JSON only. No markdown fences, no extra text."
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_keys() -> dict:
    key_path = Path(__file__).resolve().parent.parent / "key.json"
    with open(key_path) as f:
        return json.load(f)


def parse_agent_response(text: str) -> CraigslistShopAction:
    """Parse LLM JSON response into a CraigslistShopAction."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return CraigslistShopAction(
            message=text[:500],
            action_type="counter_offer",
        )

    return CraigslistShopAction(
        message=data.get("message") or "",
        action_type=data.get("action_type", "counter_offer"),
        price=data.get("price"),
    )


def print_separator():
    print(f"{DIM}{'─' * 80}{RESET}")


def print_state(obs):
    """Print current environment state."""
    print(f"\n{YELLOW}{BOLD}  ┌─ ENVIRONMENT STATE ─────────────────────────────────{RESET}")
    print(f"{YELLOW}  │ Item:              {obs.item_title}{RESET}")
    print(f"{YELLOW}  │ Category:          {obs.item_category}{RESET}")
    print(f"{YELLOW}  │ Listed price:      ${obs.listed_price:.2f}{RESET}")
    print(f"{YELLOW}  │ Current offer:     {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none'}{RESET}")
    print(f"{YELLOW}  │ Turn:              {obs.turn}{RESET}")
    print(f"{YELLOW}  └──────────────────────────────────────────────────────{RESET}")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env,
    client,
    model: str,
    strategy: str,
    episode_num: int,
    task_index=None,
    split: str = "test",
    verbose: bool = True,
    provider: str = "azure",
):
    """Run a single episode (one negotiation). Returns episode result dict."""
    system_prompt = STRATEGIES[strategy]
    # messages tracks the conversation for the Azure path; also used to build
    # claude prompts (we serialize history from messages[1:] each turn).
    messages = [{"role": "system", "content": system_prompt}]

    if verbose:
        print(f"\n{CYAN}{BOLD}{'=' * 70}{RESET}")
        print(f"{CYAN}{BOLD}  EPISODE {episode_num}  [{provider}]{RESET}")
        print(f"{CYAN}{BOLD}{'=' * 70}{RESET}")

    result = await env.reset(task_index=task_index, split=split)
    obs = result.observation

    if verbose:
        print(f"\n{YELLOW}{BOLD}  ┌─ ITEM INFO ────────────────────────────────────────────{RESET}")
        print(f"{YELLOW}  │ Title:       {obs.item_title}{RESET}")
        print(f"{YELLOW}  │ Category:    {obs.item_category}{RESET}")
        print(f"{YELLOW}  │ Listed at:   ${obs.listed_price:.2f}{RESET}")
        desc_preview = obs.item_description[:200] + "..." if len(obs.item_description) > 200 else obs.item_description
        print(f"{YELLOW}  │ Description: {desc_preview}{RESET}")
        print(f"{YELLOW}  └────────────────────────────────────────────────────────{RESET}")
        if obs.customer_message:
            print(f"\n{MAGENTA}{BOLD}  [BUYER → AGENT] (opening){RESET}")
            print(f"{MAGENTA}  {obs.customer_message}{RESET}")

    step = 0
    max_steps = 30
    cumulative_reward = 0.0

    while not result.done and step < max_steps:
        step += 1

        if verbose:
            print(f"\n{DIM}  {'─' * 60}{RESET}")
            print(f"{DIM}  Turn {obs.turn + 1}{RESET}")

        obs_parts = [
            f"Item: {obs.item_title} ({obs.item_category})",
            f"Description: {obs.item_description[:300]}",
            f"Listed price: ${obs.listed_price:.2f}",
            f"Your last offered price: {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}",
            f"Turn: {obs.turn}",
        ]
        if obs.system_message:
            obs_parts.append(f"\nSystem: {obs.system_message}")
        if obs.customer_message:
            obs_parts.append(f"\nBuyer says: {obs.customer_message}")
        obs_text = "\n".join(obs_parts)
        messages.append({"role": "user", "content": obs_text})

        if verbose:
            print(f"\n{YELLOW}{BOLD}  [ENV → AGENT] observation:{RESET}")
            for line in obs_text.split("\n"):
                print(f"{YELLOW}    {line}{RESET}")

        try:
            if provider == "claude":
                # Build a single stateless prompt with full context and call claude -p
                prompt = build_claude_prompt(system_prompt, obs, messages[1:])
                agent_text = call_claude(prompt)
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                )
                agent_text = completion.choices[0].message.content or ""
        except Exception as e:
            if verbose:
                print(f"{RED}    ERROR: {e}{RESET}")
            agent_text = json.dumps({
                "message": f"The price is ${obs.listed_price:.2f}.",
                "action_type": "counter_offer",
                "price": obs.listed_price,
            })

        messages.append({"role": "assistant", "content": agent_text})
        action = parse_agent_response(agent_text)

        if verbose:
            price_str = f" @ ${action.price:.2f}" if action.price is not None else ""
            print(f"\n{GREEN}{BOLD}  [AGENT → ENV] {action.action_type}{price_str}{RESET}")
            print(f"{GREEN}    \"{action.message}\"{RESET}")

        result = await env.step(action)
        obs = result.observation

        if result.reward and result.reward != 0:
            cumulative_reward += result.reward

        if verbose:
            if obs.customer_message:
                print(f"\n{MAGENTA}{BOLD}  [BUYER → AGENT]{RESET}")
                print(f"{MAGENTA}    \"{obs.customer_message}\"{RESET}")
            if obs.system_message:
                print(f"{CYAN}  [SYSTEM] {obs.system_message}{RESET}")
            if result.reward is not None and result.reward != 0:
                print(f"{BLUE}  [REWARD] {result.reward:.4f}{RESET}")

    outcome = obs.outcome or "?"

    if verbose:
        color = GREEN if outcome == "sold" else RED
        sale_str = f" @ ${obs.sale_price:.2f}" if outcome == "sold" else ""
        print(f"{color}    ⇒ {outcome}{sale_str}  "
              f"[{obs.item_category}]  "
              f"reward={cumulative_reward:.4f}  ({step} steps){RESET}")

    return {
        "episode": episode_num,
        "reward": cumulative_reward,
        "outcome": outcome,
        "item_title": obs.item_title,
        "item_category": obs.item_category,
        "sale_price": obs.sale_price,
        "listed_price": obs.listed_price,
        "price_retention": obs.sale_price / obs.listed_price if obs.listed_price > 0 and outcome == "sold" else 0.0,
        "turns": obs.turn,
        "total_steps": step,
        "conversation": obs.conversation_history,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run(
    base_url: str,
    strategy: str,
    num_episodes: int,
    verbose: bool,
    suffix: str | None = None,
    sample: bool = False,
    provider: str = "azure",
):
    # Azure path requires keys; Claude path does not
    if provider == "azure":
        keys = load_keys()
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=keys["azure_openai_api_key"],
            azure_endpoint=keys["azure_openai_endpoint"],
            api_version="2024-12-01-preview",
        )
        model = keys.get("azure_openai_planner_deployment", "gpt-4o")
    else:
        client = None
        model = "claude"

    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  CRAIGSLIST SHOP NEGOTIATION EVALUATION{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{DIM}  Provider:     {provider}{RESET}")
    print(f"{DIM}  Agent model:  {model}{RESET}")
    print(f"{DIM}  Strategy:     {strategy} — {STRATEGY_DESCRIPTIONS[strategy]}{RESET}")
    print(f"{DIM}  Episodes:     {num_episodes}{RESET}")
    print(f"{DIM}  Server:       {base_url}{RESET}")

    episodes = []

    # Pick task indices: sequential (0, 1, 2, ...) or random sample
    if sample:
        import random
        indices = random.sample(range(800), min(num_episodes, 800))
    else:
        indices = list(range(num_episodes))

    async with CraigslistShopEnv(base_url=base_url) as env:
        for ep, task_idx in enumerate(indices, 1):
            ep_result = await run_episode(
                env, client, model, strategy, ep,
                task_index=task_idx, split="test", verbose=verbose,
                provider=provider,
            )
            episodes.append(ep_result)

    # --- Aggregate stats ---
    rewards = [e["reward"] for e in episodes]
    sold = [e for e in episodes if e["outcome"] == "sold"]

    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    service_rate = len(sold) / len(episodes)

    sold_rewards = [e["reward"] for e in sold]
    avg_sold_reward = (
        sum(sold_rewards) / len(sold_rewards) if sold_rewards else 0.0
    )

    price_retentions = [e["price_retention"] for e in sold if e["price_retention"] > 0]
    avg_price_retention = (
        sum(price_retentions) / len(price_retentions) if price_retentions else 0.0
    )

    # Per-category breakdown
    categories = sorted(set(e["item_category"] for e in episodes))
    per_cat_stats = {}
    for cat in categories:
        cat_eps = [e for e in episodes if e["item_category"] == cat]
        cat_sold = [e for e in cat_eps if e["outcome"] == "sold"]
        cat_rewards = [e["reward"] for e in cat_eps]
        cat_retentions = [e["price_retention"] for e in cat_sold if e["price_retention"] > 0]
        per_cat_stats[cat] = {
            "count": len(cat_eps),
            "sold": len(cat_sold),
            "service_rate": len(cat_sold) / len(cat_eps) if cat_eps else 0,
            "avg_reward": sum(cat_rewards) / len(cat_rewards) if cat_rewards else 0,
            "avg_price_retention": (
                sum(cat_retentions) / len(cat_retentions) if cat_retentions else 0
            ),
        }

    # --- Print summary ---
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  RESULTS — {strategy} ({provider}) × {num_episodes} episodes{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"  Avg reward:          {avg_reward:.4f}  (std: {std_reward:.4f})")
    print(f"  Min / Max reward:    {min_reward:.4f} / {max_reward:.4f}")
    print(f"  Sale rate:           {service_rate:.0%}  ({len(sold)}/{len(episodes)})")
    print(f"  Avg price retention: {avg_price_retention:.1%}  (sold only)")
    print(f"  Avg sold reward:     {avg_sold_reward:.4f}")

    if per_cat_stats:
        print(f"\n  {BOLD}Per category:{RESET}")
        print(f"  {'Category':<18} {'N':>3}  {'Sold':>4}  {'Rate':>5}  {'AvgRwd':>7}  {'PriceRet':>8}")
        print(f"  {'─' * 52}")
        for cat, s in per_cat_stats.items():
            print(f"  {cat:<18} {s['count']:>3}  {s['sold']:>4}  "
                  f"{s['service_rate']:>4.0%}  {s['avg_reward']:>7.4f}  "
                  f"{s['avg_price_retention']:>7.1%}")

    # Per-episode table
    print(f"\n  {BOLD}Episodes:{RESET}")
    print(f"  {'#':>3}  {'Item':<30} {'Outcome':<10} {'Sale$':>8}  {'List$':>8}  {'Reward':>7}  {'Turns':>5}")
    print(f"  {'─' * 80}")
    for e in episodes:
        color = GREEN if e["outcome"] == "sold" else RED
        title = e["item_title"][:28]
        print(f"  {e['episode']:>3}  {title:<30} "
              f"{color}{e['outcome']:<10}{RESET} "
              f"${e['sale_price']:>7.2f}  ${e['listed_price']:>7.2f}  "
              f"{e['reward']:>7.4f}  {e['turns']:>5}")

    # --- Save to disk ---
    dir_name = f"runs_{suffix}" if suffix else "runs"
    runs_dir = Path(__file__).resolve().parent.parent / dir_name
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_filename = f"{strategy}_{provider}_{num_episodes}ep_{timestamp}.json"
    batch_path = runs_dir / batch_filename

    batch_data = {
        "strategy": strategy,
        "provider": provider,
        "strategy_description": STRATEGY_DESCRIPTIONS[strategy],
        "agent_model": model,
        "num_episodes": num_episodes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "service_rate": service_rate,
            "avg_price_retention": avg_price_retention,
            "avg_sold_reward": avg_sold_reward,
            "per_category": per_cat_stats,
        },
        "episodes": episodes,
    }

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    print(f"\n{GREEN}  Batch saved to: {batch_path}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM test agent for Craigslist Shop negotiation")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--provider",
        default="azure",
        choices=["azure", "claude"],
        help="LLM provider for the seller agent (default: azure)",
    )
    parser.add_argument(
        "--strategy",
        default="skilled_seller",
        choices=list(STRATEGIES.keys()),
        help="Agent strategy / system prompt to use",
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output, show only summary")
    parser.add_argument("--suffix", default=None,
                        help="Suffix for the runs directory (e.g., --suffix v2 saves to runs_v2/)")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Randomly sample episodes from the test set instead of taking the first N sequentially",
    )
    parser.add_argument("--list-strategies", action="store_true",
                        help="List available strategies and exit")
    args = parser.parse_args()

    if args.list_strategies:
        print(f"\n{BOLD}Available strategies:{RESET}\n")
        for name, desc in STRATEGY_DESCRIPTIONS.items():
            print(f"  {GREEN}{name:<22}{RESET} {desc}")
        print()
        return

    try:
        asyncio.run(run(
            args.url,
            args.strategy,
            args.episodes,
            verbose=not args.quiet,
            suffix=args.suffix,
            sample=args.sample,
            provider=args.provider,
        ))
    except ConnectionRefusedError:
        print(f"Could not connect to {args.url}. Is the server running?",
              file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")


if __name__ == "__main__":
    main()
