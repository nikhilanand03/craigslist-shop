# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the BurgerShack Environment.

Each episode is a single customer negotiation. The agent's goal is to
sell a burger at the highest price possible without the customer walking away.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BurgershackAction(Action):
    """Action the agent can take during a customer negotiation."""

    message: str = Field(
        ..., description="Natural language response to the customer"
    )
    action_type: Literal[
        "counter_offer", "accept", "reject", "serve"
    ] = Field(
        ...,
        description=(
            "Type of action: counter_offer (propose/negotiate a price), "
            "accept (agree to customer's last price), "
            "reject (refuse and end the interaction), "
            "serve (complete the sale at the stated price)"
        ),
    )
    price: float | None = Field(
        default=None,
        description=(
            "The price you are offering or accepting. Required for 'counter_offer' "
            "and 'serve'. For 'accept', defaults to the customer's last stated price."
        ),
    )


class BurgershackObservation(Observation):
    """Observation returned by the BurgerShack environment each step."""

    customer_message: str = Field(
        default="", description="The customer's current message (dialogue only)"
    )
    system_message: str = Field(
        default="",
        description="System/game messages (sale confirmations, walkaway notices)",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Full conversation history in OpenAI message format",
    )
    menu_price: float = Field(
        default=4.00, description="The listed menu price for this item"
    )
    cost_floor: float = Field(
        default=2.00,
        description="Minimum price (ingredient cost + margin). Selling below this is a loss.",
    )
    current_offer_price: float | None = Field(
        default=None,
        description="The last price explicitly offered by the agent",
    )
    turn: int = Field(default=0, description="Current negotiation turn number")
    max_turns: int = Field(
        default=10, description="Maximum turns before forced walkaway"
    )
    episode_id: str = Field(default="", description="Unique episode identifier")
    customer_persona: str = Field(
        default="",
        description="A short label for the customer type (hidden details, visible label)",
    )
    customer_willing_to_buy: bool = Field(
        default=False,
        description="Whether the customer has agreed to buy at the current offered price. "
        "The 'serve' action only succeeds when this is True.",
    )
    # Episode result fields — populated only on the final observation (done=True)
    outcome: str = Field(
        default="", description="'served' or 'walkaway' (only set when done=True)"
    )
    sale_price: float = Field(
        default=0.0, description="Final sale price (only set when done=True and served)"
    )
    profit: float = Field(
        default=0.0, description="Sale price minus cost floor (only set when done=True)"
    )
