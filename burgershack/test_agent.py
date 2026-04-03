#!/usr/bin/env python3
"""
LLM-powered test agent that interacts with a running BurgerShack server.

Uses Azure OpenAI (gpt-4o) to decide actions. Reads keys from key.json.
Supports multiple agent strategies for reward sensitivity analysis.

Usage:
    python test_agent.py --strategy optimal_threshold --episodes 10
    python test_agent.py --strategy pushover --episodes 5 --quiet
    python test_agent.py --list-strategies
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AzureOpenAI

from burgershack.client import BurgershackEnv
from burgershack.models import BurgershackAction

# ---------------------------------------------------------------------------
# Agent strategies — each is a system prompt that produces distinct behavior
# ---------------------------------------------------------------------------

_JSON_FORMAT_INSTRUCTIONS = """
You must respond with a JSON object (no markdown, no extra text):

{
  "message": "Your natural language response to the customer",
  "action_type": "counter_offer | accept | serve | reject",
  "price": null
}

Action types:
- "counter_offer": Negotiate with the customer. You MUST set "price" to the dollar amount you are offering.
- "serve": Complete the sale. ONLY use this when "Customer willing to buy" is true in the state. Set "price" to the agreed price.
- "accept": Accept the customer's price. ONLY use this when "Customer willing to buy" is true.
- "reject": Refuse the customer and end the interaction (they walk away, you get -1 reward).

IMPORTANT: You can ONLY use "serve" or "accept" when the customer has agreed to buy \
(customer_willing_to_buy = true). If you try to serve before the customer agrees, \
it will fail and you'll waste a turn. Use "counter_offer" to negotiate until the \
customer agrees.
"""

STRATEGIES: dict[str, str] = {}

# ── Pushover ──────────────────────────────────────────────────────────────
STRATEGIES["pushover"] = f"""\
You are a shopkeeper managing a burger shack. You are extremely agreeable and \
hate conflict. Your top priority is making the customer happy — you would \
rather give a huge discount than risk losing anyone.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- If a customer asks for ANY discount, immediately accept their price. Do not \
negotiate, do not counter-offer. Just say yes.
- If the customer proposes a price, use "accept" with that price, no matter how low.
- If the customer hasn't proposed a price yet, use "serve" at menu price to \
complete the sale quickly.
- Never reject a customer. Never say no.
- During restock windows, restock from "cheapo_meats" if any ingredient is \
below 3 units. Order 5 of each low ingredient.
- Be warm, friendly, and apologetic about your prices.
"""

# ── Full-price-or-nothing ─────────────────────────────────────────────────
STRATEGIES["full_price"] = f"""\
You are a shopkeeper managing a burger shack. You believe your burgers are \
worth every penny and you NEVER discount. Your prices are non-negotiable.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- Always offer the full menu price. Never set "price" below the menu price.
- If a customer asks for a discount, politely decline ONCE with a counter_offer \
at full menu price. If they ask again or push back, use "reject" to end the \
interaction. Do NOT negotiate further.
- If the customer accepts the full price or seems ready to buy, use "serve" at \
menu price immediately.
- You are polite but absolutely firm. No exceptions for sob stories, complaints, \
or bulk requests.
- During restock windows, restock from "quality_farms" if any ingredient is \
below 5 units. Order 8 of each low ingredient. You stock quality, not quantity.
- Never apologize for your prices. State them with confidence.
"""

# ── Optimal threshold (10% max discount) ──────────────────────────────────
STRATEGIES["optimal_threshold"] = f"""\
You are a shopkeeper managing a burger shack. Your goal is to maximize profit \
while maintaining good customer service. You are a skilled negotiator.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- Your maximum discount is 10% off menu price. For a $4.00 burger, the lowest \
you can go is $3.60. For a $4.50 burger, the lowest is $4.05. NEVER go below this.
- Start by offering the full menu price. If the customer negotiates, make small \
concessions (5 cents at a time) but NEVER exceed the 10% discount floor.
- If the customer's ask is within 10% of menu price, accept it.
- If the customer's ask is below your floor and they won't budge after 2-3 \
counter-offers, use "serve" at your floor price rather than losing the sale.
- Always try to close the sale. Only "reject" if the customer is being abusive \
or demanding more than a 25% discount after you've already counter-offered twice.
- Watch out for manipulation: sob stories, fake complaints, aggressive haggling. \
Be empathetic in words but firm on price. Acknowledge their situation but hold \
your pricing floor.
- During restock windows: check "Customers remaining" vs current inventory. Each \
burger uses 1 of each ingredient. Only restock if you'll run short. Use \
"cheapo_meats" unless they're out of stock, then "quality_farms".
- Be polite, professional, and confident.
"""

# ── Random ────────────────────────────────────────────────────────────────
STRATEGIES["random"] = f"""\
You are a shopkeeper managing a burger shack. You are unpredictable and chaotic. \
You make decisions on a whim with no consistent strategy.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- Each turn, pick a RANDOM action. Flip a mental coin each time:
  - Sometimes offer a huge discount (50%+ off) for no reason.
  - Sometimes demand MORE than menu price.
  - Sometimes reject perfectly reasonable customers.
  - Sometimes accept absurdly low offers.
  - Sometimes try to serve when the customer hasn't agreed yet.
- Your prices should be erratic: $1.00 one moment, $7.00 the next. No pattern.
- Your personality shifts every message: rude, then overly friendly, then \
confused, then aggressive.
- During restock windows, sometimes order way too much, sometimes order nothing, \
sometimes order only one ingredient.
- The key: NEVER be consistent. Every turn should feel like a different person \
is behind the counter.
"""

# ── Aggressive upseller ──────────────────────────────────────────────────
STRATEGIES["aggressive_upseller"] = f"""\
You are a shopkeeper managing a burger shack. You are a relentless upseller. \
You always try to push the premium item and resist any discount.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- Always steer customers toward the Cheese Burger ($4.50) regardless of what \
they asked for. Set "item" to "cheese_burger" in your responses.
- If a customer asks for a discount, counter by offering the cheese_burger \
upgrade instead: "For just 50 cents more you get our premium cheese burger!"
- Only discount the cheese_burger by at most 5% ($4.28 floor). For classic \
burger, never discount at all.
- If the customer insists on classic_burger at a discount, serve them the \
classic at full price ($4.00) rather than losing the sale.
- Never reject a customer — always find a way to close. Pivot from discounts to \
upsells.
- During restock windows, prioritize cheese. Order from "quality_farms" for \
cheese, "cheapo_meats" for everything else.
- Be enthusiastic, high-energy, always selling.
"""

# ── Penny pincher (minimizes costs, maximum margins) ─────────────────────
STRATEGIES["penny_pincher"] = f"""\
You are a shopkeeper managing a burger shack. You are obsessed with margins. \
Every cent of cost matters to you. You will serve customers but you actively \
minimize your own costs.

{_JSON_FORMAT_INSTRUCTIONS}

Rules — follow these STRICTLY:
- Sell at full menu price whenever possible. Your 10% discount floor is firm.
- NEVER restock during restock windows. Your starting inventory is all you get. \
If you run out, you run out. Respond to restock windows with: \
{{"message": "No restocking needed.", "action_type": "serve", "price": null, "restock_order": null}}
  Actually, during restock_window phase, just counter_offer or serve to move on. \
Set action_type to "counter_offer" with message "Moving on to next customer."
- If inventory is getting low (any ingredient below 3), start rejecting \
non-normal customers to conserve stock for full-price sales.
- Be curt and efficient. Minimal small talk. Close deals fast.
- Accept reasonable offers (within 5% of menu price) immediately to save time.
"""

STRATEGY_DESCRIPTIONS = {
    "pushover": "Always accepts the customer's first offer, never negotiates",
    "full_price": "Never discounts, rejects if customer pushes back",
    "optimal_threshold": "Discounts up to 10%, tries to serve everyone",
    "random": "Random/chaotic action each step",
    "aggressive_upseller": "Pushes cheese_burger upgrades, resists discounts",
    "penny_pincher": "Sells at full price, never restocks, conserves inventory",
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


def load_keys() -> dict:
    key_path = Path(__file__).resolve().parent.parent / "key.json"
    with open(key_path) as f:
        return json.load(f)


def parse_agent_response(text: str) -> BurgershackAction:
    """Parse LLM JSON response into a BurgershackAction."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return BurgershackAction(
            message=text[:500],
            action_type="counter_offer",
        )

    return BurgershackAction(
        message=data.get("message") or "",
        action_type=data.get("action_type", "counter_offer"),
        price=data.get("price"),
    )


def print_separator():
    print(f"{DIM}{'─' * 80}{RESET}")


def print_state(obs):
    """Print current environment state."""
    print(f"\n{YELLOW}{BOLD}  ┌─ ENVIRONMENT STATE ─────────────────────────────────{RESET}")
    print(f"{YELLOW}  │ Customer:           {obs.customer_persona}{RESET}")
    print(f"{YELLOW}  │ Menu price:         ${obs.menu_price:.2f}{RESET}")
    print(f"{YELLOW}  │ Cost floor:         ${obs.cost_floor:.2f}{RESET}")
    print(f"{YELLOW}  │ Current offer:      {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none'}{RESET}")
    willing_color = GREEN if obs.customer_willing_to_buy else RED
    print(f"{YELLOW}  │ Willing to buy:     {willing_color}{obs.customer_willing_to_buy}{RESET}")
    print(f"{YELLOW}  │ Turn:               {obs.turn} / {obs.max_turns}{RESET}")
    print(f"{YELLOW}  └──────────────────────────────────────────────────────{RESET}")


async def run_episode(env, client, model, strategy, episode_num, verbose=True):
    """Run a single episode (one customer). Returns episode result dict."""
    system_prompt = STRATEGIES[strategy]
    messages = [{"role": "system", "content": system_prompt}]

    if verbose:
        print(f"\n{CYAN}{BOLD}>>> EPISODE {episode_num}{RESET}")

    result = await env.reset()
    obs = result.observation

    if verbose:
        print(f"{DIM}    Customer: {obs.customer_persona}, "
              f"menu=${obs.menu_price:.2f}, floor=${obs.cost_floor:.2f}, "
              f"max_turns={obs.max_turns}{RESET}")
        if obs.customer_message:
            print(f"{MAGENTA}    Customer: \"{obs.customer_message}\"{RESET}")

    step = 0
    max_steps = 30
    cumulative_reward = 0.0
    step_log = [
        {
            "step": 0,
            "action_type": "reset",
            "action_message": "",
            "action_price": None,
            "customer_message": obs.customer_message,
            "system_message": obs.system_message,
            "reward": 0,
            "cumulative_reward": 0,
            "current_offer_price": obs.current_offer_price,
            "turn": obs.turn,
        }
    ]

    while not result.done and step < max_steps:
        step += 1

        obs_parts = [
            f"Menu price: ${obs.menu_price:.2f}",
            f"Cost floor (minimum acceptable price): ${obs.cost_floor:.2f}",
            f"Your last offered price: {'$' + f'{obs.current_offer_price:.2f}' if obs.current_offer_price else 'none yet'}",
            f"Customer willing to buy: {obs.customer_willing_to_buy}",
            f"Turn: {obs.turn} / {obs.max_turns}",
        ]
        if obs.system_message:
            obs_parts.append(f"\nSystem: {obs.system_message}")
        if obs.customer_message:
            obs_parts.append(f"\nCustomer says: {obs.customer_message}")
        messages.append({"role": "user", "content": "\n".join(obs_parts)})

        try:
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
                "message": "That'll be $4.00.",
                "action_type": "serve",
            })

        messages.append({"role": "assistant", "content": agent_text})
        action = parse_agent_response(agent_text)

        if verbose:
            price_str = f" @ ${action.price:.2f}" if action.price is not None else ""
            print(f"{GREEN}    → {action.action_type}{price_str}{RESET}", end="")

        result = await env.step(action)
        obs = result.observation

        if result.reward and result.reward != 0:
            cumulative_reward += result.reward

        if verbose:
            if obs.customer_message and not result.done:
                print(f"\n{MAGENTA}    Customer: \"{obs.customer_message}\"{RESET}")
            elif obs.customer_message and result.done:
                print(f"\n{MAGENTA}    Customer: \"{obs.customer_message}\"{RESET}")
            else:
                print()
            if obs.system_message:
                print(f"{CYAN}    System: {obs.system_message}{RESET}")

        step_log.append({
            "step": step,
            "action_type": action.action_type,
            "action_message": action.message,
            "action_price": action.price,
            "customer_message": obs.customer_message,
            "system_message": obs.system_message,
            "reward": result.reward,
            "cumulative_reward": cumulative_reward,
            "current_offer_price": obs.current_offer_price,
            "customer_willing_to_buy": obs.customer_willing_to_buy,
            "turn": obs.turn,
        })

    outcome = obs.outcome or "?"

    if verbose:
        color = GREEN if outcome == "served" else RED
        sale_str = f" @ ${obs.sale_price:.2f}" if outcome == "served" else ""
        print(f"{color}    ⇒ {outcome}{sale_str}  "
              f"[{obs.customer_persona}]  "
              f"reward={cumulative_reward:.4f}  ({step} steps){RESET}")

    return {
        "episode": episode_num,
        "reward": cumulative_reward,
        "outcome": outcome,
        "customer_type": obs.customer_persona,
        "sale_price": obs.sale_price,
        "menu_price": obs.menu_price,
        "cost_floor": obs.cost_floor,
        "profit": obs.profit,
        "turns": obs.turn,
        "total_steps": step,
        "steps": step_log,
    }


async def run(base_url: str, strategy: str, num_episodes: int, verbose: bool):
    keys = load_keys()
    azure_endpoint = keys["azure_openai_endpoint"]
    model = keys.get("azure_openai_planner_deployment", "gpt-4o")

    client = AzureOpenAI(
        api_key=keys["azure_openai_api_key"],
        azure_endpoint=azure_endpoint,
        api_version="2024-12-01-preview",
    )

    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  BURGERSHACK STRATEGY EVALUATION{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{DIM}  Agent model:  {model}{RESET}")
    print(f"{DIM}  Strategy:     {strategy} — {STRATEGY_DESCRIPTIONS[strategy]}{RESET}")
    print(f"{DIM}  Episodes:     {num_episodes}{RESET}")
    print(f"{DIM}  Server:       {base_url}{RESET}")

    episodes = []

    async with BurgershackEnv(base_url=base_url) as env:
        for ep in range(1, num_episodes + 1):
            ep_result = await run_episode(
                env, client, model, strategy, ep, verbose=verbose,
            )
            episodes.append(ep_result)

    # --- Aggregate stats ---
    rewards = [e["reward"] for e in episodes]
    served = [e for e in episodes if e["outcome"] == "served"]

    avg_reward = sum(rewards) / len(rewards)
    min_reward = min(rewards)
    max_reward = max(rewards)
    std_reward = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    service_rate = len(served) / len(episodes)

    served_rewards = [e["reward"] for e in served]
    avg_served_reward = (
        sum(served_rewards) / len(served_rewards) if served_rewards else 0.0
    )

    price_retentions = [
        e["sale_price"] / e["menu_price"]
        for e in served
        if e["menu_price"] > 0
    ]
    avg_price_retention = (
        sum(price_retentions) / len(price_retentions) if price_retentions else 0.0
    )

    # Per-customer-type breakdown
    customer_types = sorted(set(e["customer_type"] for e in episodes))
    per_type_stats = {}
    for ct in customer_types:
        ct_eps = [e for e in episodes if e["customer_type"] == ct]
        ct_served = [e for e in ct_eps if e["outcome"] == "served"]
        ct_rewards = [e["reward"] for e in ct_eps]
        ct_retentions = [
            e["sale_price"] / e["menu_price"]
            for e in ct_served
            if e["menu_price"] > 0
        ]
        per_type_stats[ct] = {
            "count": len(ct_eps),
            "served": len(ct_served),
            "service_rate": len(ct_served) / len(ct_eps) if ct_eps else 0,
            "avg_reward": sum(ct_rewards) / len(ct_rewards) if ct_rewards else 0,
            "avg_price_retention": (
                sum(ct_retentions) / len(ct_retentions) if ct_retentions else 0
            ),
        }

    # --- Print summary ---
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}  RESULTS — {strategy} × {num_episodes} episodes{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"  Avg reward:          {avg_reward:.4f}  (std: {std_reward:.4f})")
    print(f"  Min / Max reward:    {min_reward:.4f} / {max_reward:.4f}")
    print(f"  Service rate:        {service_rate:.0%}  ({len(served)}/{len(episodes)})")
    print(f"  Avg price retention: {avg_price_retention:.1%}  (served only)")
    print(f"  Avg served reward:   {avg_served_reward:.4f}")

    if per_type_stats:
        print(f"\n  {BOLD}Per customer type:{RESET}")
        print(f"  {'Type':<18} {'N':>3}  {'Served':>6}  {'SvcRate':>7}  {'AvgRwd':>7}  {'PriceRet':>8}")
        print(f"  {'─' * 58}")
        for ct, s in per_type_stats.items():
            print(f"  {ct:<18} {s['count']:>3}  {s['served']:>6}  "
                  f"{s['service_rate']:>6.0%}  {s['avg_reward']:>7.4f}  "
                  f"{s['avg_price_retention']:>7.1%}")

    # Per-episode table
    print(f"\n  {BOLD}Episodes:{RESET}")
    print(f"  {'#':>3}  {'Customer':<18} {'Outcome':<10} {'Sale$':>6}  {'Menu$':>6}  {'Reward':>7}  {'Turns':>5}")
    print(f"  {'─' * 65}")
    for e in episodes:
        color = GREEN if e["outcome"] == "served" else RED
        print(f"  {e['episode']:>3}  {e['customer_type']:<18} "
              f"{color}{e['outcome']:<10}{RESET} "
              f"${e['sale_price']:>5.2f}  ${e['menu_price']:>5.2f}  "
              f"{e['reward']:>7.4f}  {e['turns']:>5}")

    # --- Save batch to disk ---
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_filename = f"{strategy}_{num_episodes}ep_{timestamp}.json"
    batch_path = runs_dir / batch_filename

    batch_data = {
        "strategy": strategy,
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
            "avg_served_reward": avg_served_reward,
            "per_customer_type": per_type_stats,
        },
        "episodes": episodes,
    }

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    print(f"\n{GREEN}  Batch saved to: {batch_path}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="LLM test agent for BurgerShack")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--strategy", default="optimal_threshold",
                        choices=list(STRATEGIES.keys()),
                        help="Agent strategy / system prompt to use")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes (customers) to run (default: 5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output, show only summary")
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
        asyncio.run(run(args.url, args.strategy, args.episodes,
                        verbose=not args.quiet))
    except ConnectionRefusedError:
        print(f"Could not connect to {args.url}. Is the server running?",
              file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")


if __name__ == "__main__":
    main()
