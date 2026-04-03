# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BurgerShack Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BurgershackAction, BurgershackObservation


class BurgershackEnv(
    EnvClient[BurgershackAction, BurgershackObservation, State]
):
    """
    Client for the BurgerShack Environment.

    Each episode is a single customer negotiation over WebSocket.

    Example:
        >>> async with BurgershackEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     print(result.observation.customer_message)
        ...     action = BurgershackAction(
        ...         message="That'll be $4.00",
        ...         action_type="serve",
        ...         price=4.00,
        ...     )
        ...     result = await env.step(action)
    """

    def _step_payload(self, action: BurgershackAction) -> Dict:
        payload = {
            "message": action.message,
            "action_type": action.action_type,
        }
        if action.price is not None:
            payload["price"] = action.price
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[BurgershackObservation]:
        obs_data = payload.get("observation", {})
        observation = BurgershackObservation(
            customer_message=obs_data.get("customer_message", ""),
            system_message=obs_data.get("system_message", ""),
            conversation_history=obs_data.get("conversation_history", []),
            menu_price=obs_data.get("menu_price", 4.00),
            cost_floor=obs_data.get("cost_floor", 2.00),
            current_offer_price=obs_data.get("current_offer_price"),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 10),
            episode_id=obs_data.get("episode_id", ""),
            customer_persona=obs_data.get("customer_persona", ""),
            customer_willing_to_buy=obs_data.get("customer_willing_to_buy", False),
            outcome=obs_data.get("outcome", ""),
            sale_price=obs_data.get("sale_price", 0.0),
            profit=obs_data.get("profit", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
