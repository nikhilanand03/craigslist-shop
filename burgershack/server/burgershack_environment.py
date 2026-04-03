# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
BurgerShack Environment Implementation.

Each episode is a single customer negotiation. A random customer persona is
sampled from the population. The agent's goal: sell the burger at the highest
price possible without the customer walking away.

Reward:
  served  →  (sale_price - cost_floor) / (menu_price - cost_floor)   ∈ [0, 1]
  walkaway → -1.0
"""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from openai import AzureOpenAI

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BurgershackAction, BurgershackObservation
except ImportError:
    from models import BurgershackAction, BurgershackObservation

from .scoring import compute_reward
from .state_machine import Phase, StateMachine

# Paths
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_CUSTOMERS_DIR = _PACKAGE_DIR / "customers"
_CONFIG_DIR = _PACKAGE_DIR / "config"

# Fixed economics
MENU_PRICE = 4.00
INGREDIENT_COST = 1.75
MARGIN = 0.25
COST_FLOOR = INGREDIENT_COST + MARGIN  # $2.00 — minimum acceptable price

# Max turns before forced walkaway (per-customer override takes precedence)
DEFAULT_MAX_TURNS = 10


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_customer_personas() -> dict[str, dict]:
    personas = {}
    for yaml_file in _CUSTOMERS_DIR.glob("*.yaml"):
        data = _load_yaml(yaml_file)
        personas[data["type"]] = data
    return personas


class BurgerShackEnvironment(Environment):
    """
    BurgerShack benchmark environment.

    Each episode: a random customer walks up. The agent negotiates.
    Episode ends when the agent serves/accepts (sale) or the customer
    walks away (no sale).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._personas = _load_customer_personas()
        self._persona_types = list(self._personas.keys())

        # Per-episode state (set on reset)
        self._rng: np.random.Generator = np.random.default_rng()
        self._sm: StateMachine | None = None
        self._current_customer: dict | None = None
        self._conversation: list[dict] = []
        self._negotiation_turn: int = 0
        self._current_offer_price: float | None = None
        self._terminated: bool = False
        self._max_turns: int = DEFAULT_MAX_TURNS
        self._customer_willing_to_buy: bool = False

        # Customer LLM client
        self._customer_llm: AzureOpenAI | None = None
        self._customer_model: str = "gpt-4o"

    # ── Key loading ──────────────────────────────────────────────────────

    def _load_keys(self) -> dict:
        search_paths = [
            _PACKAGE_DIR.parent / "key.json",
            _PACKAGE_DIR / "key.json",
            Path("/app/env/key.json"),
            Path("/app/key.json"),
        ]
        for key_path in search_paths:
            if key_path.exists():
                with open(key_path) as f:
                    return json.load(f)
        return {}

    def _init_customer_llm(self) -> None:
        keys = self._load_keys()
        api_key = keys.get("azure_openai_api_key", os.environ.get("AZURE_OPENAI_API_KEY", ""))
        endpoint = keys.get("azure_openai_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
        if api_key and endpoint:
            self._customer_llm = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version="2024-12-01-preview",
            )
            model_key = keys.get("azure_openai_htmlgen_deployment")
            if model_key:
                self._customer_model = model_key

    # ── Customer LLM ─────────────────────────────────────────────────────

    def _get_customer_response(self, agent_message: str) -> tuple[str, bool]:
        """
        Call the customer LLM to generate a response.

        Returns (message, willing_to_buy). The willing_to_buy flag indicates
        whether the customer accepts the current price and is ready to complete
        the purchase. The agent can only successfully 'serve' when this is True.
        """
        if not self._current_customer:
            return "", False

        system_prompt = self._current_customer.get("system_prompt", "You are a customer.")
        target = self._current_customer.get("target_price", MENU_PRICE)
        threshold = self._current_customer.get("walkaway_threshold", 0.30)
        max_acceptable = target * (1 + threshold)

        context = (
            f"\n\nContext: The menu price for a burger is ${MENU_PRICE:.2f}. "
            f"You are interacting with the shopkeeper.\n\n"
            f"You MUST respond with a JSON object:\n"
            f'{{"message": "your response (1-3 sentences)", '
            f'"willing_to_buy": true or false}}\n\n'
            f"Set willing_to_buy to true ONLY if the shopkeeper has offered a "
            f"price you find acceptable (at or below ${max_acceptable:.2f}) and "
            f"you want to complete the purchase. Otherwise set it to false.\n"
            f"Respond with valid JSON only, no extra text."
        )

        messages = [{"role": "system", "content": system_prompt + context}]
        for msg in self._conversation:
            messages.append(msg)
        messages.append({"role": "user", "content": agent_message})

        if self._customer_llm:
            try:
                resp = self._customer_llm.chat.completions.create(
                    model=self._customer_model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                )
                raw = resp.choices[0].message.content or ""
                return self._parse_customer_response(raw)
            except Exception as e:
                print(f"[BurgerShack] Customer LLM error: {e}", flush=True)
                return self._fallback_customer_response()
        else:
            return self._fallback_customer_response()

    def _parse_customer_response(self, raw: str) -> tuple[str, bool]:
        """Parse the customer LLM's JSON response into (message, willing_to_buy)."""
        text = raw.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            message = data.get("message", "")
            willing = bool(data.get("willing_to_buy", False))
            return message, willing
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, treat the raw text as the message
            # and infer willingness from keywords
            willing = any(
                phrase in raw.lower()
                for phrase in ["i'll take it", "deal", "sounds good", "yes", "i accept", "let's do it"]
            )
            return raw, willing

    def _fallback_customer_response(self) -> tuple[str, bool]:
        target = self._current_customer.get("target_price", MENU_PRICE) if self._current_customer else MENU_PRICE
        return f"Hmm, I was hoping to pay around ${target:.2f}. Can you do that?", False

    # ── Observation builder ──────────────────────────────────────────────

    def _make_observation(self, customer_msg: str = "", system_msg: str = "",
                          reward: float = 0.0, done: bool = False,
                          info: dict | None = None) -> BurgershackObservation:
        info = info or {}
        ctype = self._current_customer.get("type", "") if self._current_customer else ""
        return BurgershackObservation(
            customer_message=customer_msg,
            system_message=system_msg,
            conversation_history=list(self._conversation),
            menu_price=MENU_PRICE,
            cost_floor=COST_FLOOR,
            current_offer_price=self._current_offer_price,
            turn=self._negotiation_turn,
            max_turns=self._max_turns,
            episode_id=self._state.episode_id,
            customer_persona=ctype,
            customer_willing_to_buy=self._customer_willing_to_buy,
            outcome=info.get("outcome", ""),
            sale_price=info.get("sale_price", 0.0),
            profit=info.get("profit", 0.0),
            done=done,
            reward=reward,
            metadata=info,
        )

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              **kwargs) -> BurgershackObservation:
        """Reset: sample a random customer and start negotiation."""
        self._rng = np.random.default_rng(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sm = StateMachine()
        self._terminated = False
        self._negotiation_turn = 0
        self._current_offer_price = None
        self._customer_willing_to_buy = False
        self._conversation = []

        # Initialize customer LLM on first reset
        if self._customer_llm is None:
            self._init_customer_llm()

        # Sample a random customer persona uniformly from the full population
        ctype = self._rng.choice(self._persona_types)
        self._current_customer = dict(self._personas[ctype])
        self._max_turns = self._current_customer.get("max_turns", DEFAULT_MAX_TURNS)

        # Transition and get customer opening
        self._sm.transition(Phase.NEGOTIATION)
        opening_msg, _ = self._get_customer_response(
            "You just walked up to a burger shack. Start the conversation — "
            "say what you want or begin your tactic."
        )
        # Customer is never willing_to_buy on the opening message
        self._customer_willing_to_buy = False
        self._conversation.append({"role": "assistant", "content": opening_msg})

        return self._make_observation(customer_msg=opening_msg)

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: BurgershackAction) -> BurgershackObservation:
        if self._terminated:
            return self._make_observation(
                system_msg="Episode is over.", done=True
            )

        self._state.step_count += 1
        action_type = action.action_type

        if action_type == "counter_offer":
            return self._handle_counter_offer(action)
        elif action_type == "accept":
            return self._handle_accept(action)
        elif action_type == "serve":
            return self._handle_serve(action)
        elif action_type == "reject":
            return self._handle_reject(action)
        else:
            return self._make_observation(
                system_msg=f"Unknown action type: {action_type}"
            )

    # ── Action handlers ──────────────────────────────────────────────────

    def _handle_counter_offer(self, action: BurgershackAction) -> BurgershackObservation:
        self._negotiation_turn += 1

        # Track offered price
        if action.price is not None:
            self._current_offer_price = action.price

        self._conversation.append({"role": "user", "content": action.message})

        # Don't check walkaway on the first 2 turns — give negotiation a chance
        if self._negotiation_turn >= 3 and self._should_customer_walk():
            farewell_msg, _ = self._get_customer_response(
                "The price is too high for you. Say a brief goodbye and leave."
            )
            return self._handle_walkaway(farewell_msg)

        # Check max turns
        if self._negotiation_turn >= self._max_turns:
            farewell_msg, _ = self._get_customer_response(
                "You've been negotiating too long and are frustrated. "
                "Say a brief goodbye and leave."
            )
            return self._handle_walkaway(farewell_msg)

        # Get customer response (with structured willing_to_buy)
        customer_reply, willing = self._get_customer_response(action.message)
        self._customer_willing_to_buy = willing
        self._conversation.append({"role": "assistant", "content": customer_reply})

        return self._make_observation(customer_msg=customer_reply)

    def _handle_accept(self, action: BurgershackAction) -> BurgershackObservation:
        """Agent accepts the customer's price. Only works if customer is willing to buy."""
        self._conversation.append({"role": "user", "content": action.message})

        if not self._customer_willing_to_buy:
            return self._make_observation(
                system_msg="Cannot accept — the customer hasn't agreed to a price yet. "
                "Continue negotiating with counter_offer.",
            )

        if action.price is not None:
            sale_price = action.price
        elif self._current_offer_price is not None:
            sale_price = self._current_offer_price
        else:
            sale_price = MENU_PRICE

        return self._complete_sale(sale_price)

    def _handle_serve(self, action: BurgershackAction) -> BurgershackObservation:
        """Serve the customer. Only works if customer has agreed to buy."""
        self._conversation.append({"role": "user", "content": action.message})

        if not self._customer_willing_to_buy:
            return self._make_observation(
                system_msg="Cannot serve — the customer hasn't agreed to buy yet. "
                "Continue negotiating with counter_offer until the customer "
                "indicates they are willing to buy (customer_willing_to_buy = true).",
            )

        sale_price = action.price if action.price is not None else MENU_PRICE
        return self._complete_sale(sale_price)

    def _handle_reject(self, action: BurgershackAction) -> BurgershackObservation:
        self._conversation.append({"role": "user", "content": action.message})
        farewell_msg, _ = self._get_customer_response(
            "The shopkeeper has rejected your request. "
            "Say a brief disappointed goodbye and leave."
        )
        return self._handle_walkaway(farewell_msg)

    # ── Walkaway logic ───────────────────────────────────────────────────

    def _should_customer_walk(self) -> bool:
        if not self._current_customer or self._current_offer_price is None:
            return False
        target = self._current_customer.get("target_price", MENU_PRICE)
        threshold = self._current_customer.get("walkaway_threshold", 0.30)
        max_acceptable = target * (1 + threshold)
        return self._current_offer_price > max_acceptable

    def _handle_walkaway(self, farewell: str) -> BurgershackObservation:
        self._terminated = True
        self._conversation.append({"role": "assistant", "content": farewell})
        self._sm.transition(Phase.WALKAWAY)

        reward, info = compute_reward(
            sale_price=0,
            menu_price=MENU_PRICE,
            cost_floor=COST_FLOOR,
            served=False,
        )
        info["customer_type"] = self._current_customer.get("type", "unknown")
        info["turns"] = self._negotiation_turn
        info["outcome"] = "walkaway"

        return self._make_observation(
            customer_msg=farewell,
            system_msg="Customer walked away. No sale.",
            reward=reward,
            done=True,
            info=info,
        )

    # ── Sale completion ──────────────────────────────────────────────────

    def _complete_sale(self, sale_price: float) -> BurgershackObservation:
        self._terminated = True
        self._sm.transition(Phase.TRANSACTION)

        quantity = self._current_customer.get("quantity", 1) if self._current_customer else 1
        total_sale = sale_price * quantity
        total_menu = MENU_PRICE * quantity
        total_floor = COST_FLOOR * quantity

        reward, info = compute_reward(
            sale_price=total_sale,
            menu_price=total_menu,
            cost_floor=total_floor,
            served=True,
        )
        info["customer_type"] = self._current_customer.get("type", "unknown")
        info["turns"] = self._negotiation_turn
        info["outcome"] = "served"
        info["quantity"] = quantity

        return self._make_observation(
            system_msg=f"Sale complete! Sold {quantity}x burger for ${total_sale:.2f}.",
            reward=reward,
            done=True,
            info=info,
        )

    # ── State property ───────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state
