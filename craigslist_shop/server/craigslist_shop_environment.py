"""
Craigslist Shop Negotiation Environment.

Each episode is a single customer negotiation over a Craigslist-style item.
A task (item + customer persona) is sampled from tasks/train.json. The agent
(seller) tries to sell the item as close to the listed price as possible.

The customer LLM role-plays the buyer using the persona's system_prompt.
It decides autonomously when to walk away ([WALKAWAY]) or accept ([ACCEPT $X.XX]).

Reward:
  sold     → sale_price / listed_price   ∈ [0, 1+]
  walkaway → 0.0
"""

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
from openai import AzureOpenAI, OpenAI

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CraigslistShopAction, CraigslistShopObservation
except ImportError:
    from models import CraigslistShopAction, CraigslistShopObservation

from .scoring import compute_reward
from .state_machine import Phase, StateMachine

# Paths
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_TASKS_DIR = _PACKAGE_DIR / "tasks"

# Max turns as a hard safety cap (customer LLM decides walkaway before this)
MAX_TURNS = 20

# Regex patterns for parsing customer LLM output
_WALKAWAY_RE = re.compile(r"\[WALKAWAY\]", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"\[ACCEPT\s*\$?([\d]+\.?\d*)\]", re.IGNORECASE)


def _load_tasks(split: str = "train") -> list[dict]:
    """Load tasks from tasks/{split}.json."""
    path = _TASKS_DIR / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tasks file not found: {path}. Run extract_personas.py first."
        )
    with open(path) as f:
        return json.load(f)


class CraigslistShopEnvironment(Environment):
    """
    Craigslist-style negotiation environment.

    Each episode: a task is sampled (item + buyer persona). The agent
    negotiates as the seller. Episode ends when the customer walks away
    or accepts a price.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Load both task pools
        self._task_splits: dict[str, list[dict]] = {}
        for split in ("train", "test"):
            try:
                self._task_splits[split] = _load_tasks(split)
            except FileNotFoundError:
                print(f"[CraigslistShop] WARNING: No tasks/{split}.json found. "
                      "Run extract_personas.py first.", flush=True)
                self._task_splits[split] = []
        self._tasks = self._task_splits.get("train", [])

        # Pre-compute difficulty buckets from train tasks.
        # Difficulty is based on buyer_target_price / listed_price ratio:
        #   easy   >= 0.80  (buyer willing to pay near list price)
        #   medium  0.65–0.80
        #   hard   < 0.65  (aggressive lowballer)
        def _ratio(t: dict) -> float:
            listed = t["item"].get("listed_price", 0)
            target = t["item"].get("buyer_target_price", 0)
            return target / listed if listed > 0 else 0.0

        train_tasks = self._task_splits.get("train", [])
        self._task_splits["easy"] = [t for t in train_tasks if _ratio(t) >= 0.80]
        self._task_splits["medium"] = [t for t in train_tasks if 0.65 <= _ratio(t) < 0.80]
        self._task_splits["hard"] = [t for t in train_tasks if _ratio(t) < 0.65]

        # Per-episode state (set on reset)
        self._rng: np.random.Generator = np.random.default_rng()
        self._sm: StateMachine | None = None
        self._current_task: dict | None = None
        self._conversation: list[dict] = []
        self._negotiation_turn: int = 0
        self._current_offer_price: float | None = None
        self._terminated: bool = False

        # Customer LLM client (AzureOpenAI or OpenAI)
        self._customer_llm: AzureOpenAI | OpenAI | None = None
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

        # Option 1: Standard OpenAI
        openai_key = keys.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        if openai_key:
            self._customer_llm = OpenAI(api_key=openai_key)
            self._customer_model = keys.get("openai_model", os.environ.get("OPENAI_MODEL", "gpt-4o"))
            return

        # Option 2: Azure OpenAI
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

    def _build_few_shot_context(self) -> str:
        """Format the source conversation as few-shot examples for the customer LLM."""
        source_conv = self._current_task.get("source_conversation", [])
        if not source_conv:
            return ""

        lines = ["\nHere is a real conversation you had in the past for a similar item. "
                 "Use this as a reference for your tone, style, and negotiation approach "
                 "(but adapt to the current situation — the seller may behave differently):\n"]
        for turn in source_conv:
            role = turn["role"].upper()
            lines.append(f"  {role}: {turn['message']}")

        return "\n".join(lines)

    def _get_customer_response(self, agent_message: str) -> str:
        """
        Call the customer LLM to generate a response.

        Returns the raw message text. The caller parses [WALKAWAY] / [ACCEPT]
        tags from it.
        """
        if not self._current_task:
            return ""

        system_prompt = self._current_task["persona"].get("system_prompt", "You are a buyer.")
        item = self._current_task["item"]

        # Add item context
        context = (
            f"\n\nITEM YOU ARE NEGOTIATING FOR:\n"
            f"  Title: {item['title']}\n"
            f"  Category: {item['category']}\n"
            f"  Listed price: ${item['listed_price']}\n"
        )

        # Add source conversation as few-shot reference
        few_shot = self._build_few_shot_context()

        messages = [{"role": "system", "content": system_prompt + context + few_shot}]
        for msg in self._conversation:
            messages.append(msg)
        messages.append({"role": "user", "content": agent_message})

        if self._customer_llm:
            try:
                resp = self._customer_llm.chat.completions.create(
                    model=self._customer_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"[CraigslistShop] Customer LLM error: {e}", flush=True)
                return self._fallback_customer_response()
        else:
            return self._fallback_customer_response()

    def _fallback_customer_response(self) -> str:
        """Fallback when LLM is unavailable."""
        if self._current_task:
            target = self._current_task["item"].get("buyer_target_price", 0)
            return f"I was hoping to pay around ${target:.2f}. Can you do that?"
        return "What's your best price?"

    def _parse_customer_tags(self, message: str) -> tuple[str, str | None, float | None]:
        """
        Parse customer message for action tags.

        Returns:
            (clean_message, action, price)
            action is "walkaway", "accept", or None
            price is the accepted price (if action == "accept") or None
        """
        # Check for walkaway
        if _WALKAWAY_RE.search(message):
            clean = _WALKAWAY_RE.sub("", message).strip()
            return clean, "walkaway", None

        # Check for accept
        match = _ACCEPT_RE.search(message)
        if match:
            price = float(match.group(1))
            clean = _ACCEPT_RE.sub("", message).strip()
            return clean, "accept", price

        return message, None, None

    # ── Observation builder ──────────────────────────────────────────────

    def _make_observation(self, customer_msg: str = "", system_msg: str = "",
                          reward: float = 0.0, done: bool = False,
                          info: dict | None = None) -> CraigslistShopObservation:
        info = info or {}
        item = self._current_task.get("item", {}) if self._current_task else {}
        return CraigslistShopObservation(
            customer_message=customer_msg,
            system_message=system_msg,
            conversation_history=list(self._conversation),
            item_category=item.get("category", ""),
            item_title=item.get("title", ""),
            item_description=item.get("description", ""),
            listed_price=item.get("listed_price", 0.0),
            current_offer_price=self._current_offer_price,
            turn=self._negotiation_turn,
            episode_id=self._state.episode_id,
            outcome=info.get("outcome", ""),
            sale_price=info.get("sale_price", 0.0),
            profit=info.get("price_retention", 0.0),
            done=done,
            reward=reward,
            metadata=info,
        )

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              task_index: int | None = None, split: str | None = None,
              **kwargs) -> CraigslistShopObservation:
        """
        Reset: sample a task and start negotiation.

        Args:
            seed: Random seed for reproducibility
            task_index: Specific task index to use (for evaluation). If None,
                        a random task is sampled.
            split: Which task split to use ("train" or "test"). Defaults to "train".
        """
        self._rng = np.random.default_rng(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._sm = StateMachine()
        self._terminated = False
        self._negotiation_turn = 0
        self._current_offer_price = None
        self._conversation = []

        # Initialize customer LLM on first reset
        if self._customer_llm is None:
            self._init_customer_llm()

        # Select task pool based on split
        active_split = split or "train"
        tasks = self._task_splits.get(active_split, self._task_splits.get("train", []))

        if not tasks:
            return self._make_observation(
                system_msg=f"ERROR: No tasks loaded for split '{active_split}'. "
                           "Run extract_personas.py first."
            )

        if task_index is not None:
            idx = task_index % len(tasks)
        else:
            idx = int(self._rng.integers(0, len(tasks)))
        self._current_task = tasks[idx]

        item = self._current_task["item"]

        # Transition to negotiation and get customer opening
        self._sm.transition(Phase.NEGOTIATION)
        opening = self._get_customer_response(
            f"You see a listing for '{item['title']}' priced at ${item['listed_price']}. "
            f"Start the conversation — express your interest or begin negotiating."
        )

        # Parse tags from opening (shouldn't have any, but be safe)
        clean_opening, action, price = self._parse_customer_tags(opening)
        self._conversation.append({"role": "assistant", "content": clean_opening})

        return self._make_observation(customer_msg=clean_opening)

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: CraigslistShopAction) -> CraigslistShopObservation:
        """
        Process one turn. The agent sends a message (and optional price).
        The customer LLM responds and may [ACCEPT] or [WALKAWAY].
        """
        if self._terminated:
            return self._make_observation(
                system_msg="Episode is over.", done=True
            )

        self._state.step_count += 1
        self._negotiation_turn += 1

        if action.price is not None:
            self._current_offer_price = action.price

        self._conversation.append({"role": "user", "content": action.message})

        # Hard safety cap on turns
        if self._negotiation_turn >= MAX_TURNS:
            return self._handle_walkaway(
                "I've been going back and forth too long. I'm going to pass. Thanks anyway."
            )

        # Get customer response and parse for tags
        raw_response = self._get_customer_response(action.message)
        clean_msg, customer_action, accepted_price = self._parse_customer_tags(raw_response)

        self._conversation.append({"role": "assistant", "content": clean_msg})

        if customer_action == "walkaway":
            return self._handle_walkaway(clean_msg)
        elif customer_action == "accept" and accepted_price is not None:
            return self._complete_sale(accepted_price, clean_msg)

        return self._make_observation(customer_msg=clean_msg)

    # ── Walkaway ─────────────────────────────────────────────────────────

    def _handle_walkaway(self, farewell: str) -> CraigslistShopObservation:
        self._terminated = True
        if farewell not in [msg["content"] for msg in self._conversation]:
            self._conversation.append({"role": "assistant", "content": farewell})
        self._sm.transition(Phase.WALKAWAY)

        item = self._current_task.get("item", {}) if self._current_task else {}
        listed_price = item.get("listed_price", 0.0)

        reward, info = compute_reward(
            sale_price=0,
            listed_price=listed_price,
            served=False,
        )
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

    def _complete_sale(self, sale_price: float, customer_msg: str = "") -> CraigslistShopObservation:
        self._terminated = True
        self._sm.transition(Phase.TRANSACTION)

        item = self._current_task.get("item", {}) if self._current_task else {}
        listed_price = item.get("listed_price", 0.0)

        reward, info = compute_reward(
            sale_price=sale_price,
            listed_price=listed_price,
            served=True,
        )
        info["turns"] = self._negotiation_turn
        info["outcome"] = "sold"

        return self._make_observation(
            customer_msg=customer_msg,
            system_msg=f"Sale complete! Sold for ${sale_price:.2f} "
                       f"(listed at ${listed_price:.2f}, "
                       f"retention: {reward:.0%}).",
            reward=reward,
            done=True,
            info=info,
        )

    # ── State property ───────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state
