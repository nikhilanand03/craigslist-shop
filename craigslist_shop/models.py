"""
Data models for the Craigslist Shop Negotiation Environment.

Each episode is a single customer negotiation over a Craigslist-style item.
The agent (seller) tries to sell the item as close to the listed price as
possible. The customer (LLM) negotiates based on an extracted persona.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CraigslistShopAction(Action):
    """Action the agent (seller) sends each turn: a message and an optional price."""

    message: str = Field(
        ..., description="Natural language response to the customer"
    )
    price: float | None = Field(
        default=None,
        description="The price you are offering. Include this when making or adjusting an offer.",
    )


class CraigslistShopObservation(Observation):
    """Observation returned by the environment each step."""

    # ── Customer dialogue ───────────────────────────────────────────────
    customer_message: str = Field(
        default="", description="The customer's current message"
    )
    system_message: str = Field(
        default="",
        description="System messages (sale confirmations, walkaway notices)",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Full conversation history in OpenAI message format",
    )

    # ── Item info (given to the agent at the start of each episode) ────
    item_category: str = Field(
        default="", description="Item category (e.g., phone, furniture, bike)"
    )
    item_title: str = Field(
        default="", description="Item listing title"
    )
    item_description: str = Field(
        default="", description="Item listing description"
    )
    listed_price: float = Field(
        default=0.0, description="The listed asking price for this item"
    )

    # ── Negotiation state ───────────────────────────────────────────────
    current_offer_price: float | None = Field(
        default=None,
        description="The last price explicitly offered by the agent (seller)",
    )
    turn: int = Field(default=0, description="Current negotiation turn number")
    episode_id: str = Field(default="", description="Unique episode identifier")

    # ── Episode result (populated only when done=True) ──────────────────
    outcome: str = Field(
        default="", description="'sold' or 'walkaway' (only set when done=True)"
    )
    sale_price: float = Field(
        default=0.0, description="Final sale price (only set when done=True and sold)"
    )
    profit: float = Field(
        default=0.0,
        description="How much of the listed price was retained (sale_price / listed_price)",
    )
