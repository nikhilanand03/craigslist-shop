"""Craigslist Shop Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CraigslistShopAction, CraigslistShopObservation


class CraigslistShopEnv(
    EnvClient[CraigslistShopAction, CraigslistShopObservation, State]
):
    """
    Client for the Craigslist Shop Negotiation Environment.

    Each episode is a single customer negotiation over WebSocket.

    Example:
        >>> async with CraigslistShopEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     print(result.observation.customer_message)
        ...     action = CraigslistShopAction(
        ...         message="I can do $150 for this.",
        ...         action_type="counter_offer",
        ...         price=150.00,
        ...     )
        ...     result = await env.step(action)
    """

    def _step_payload(self, action: CraigslistShopAction) -> Dict:
        payload = {"message": action.message}
        if action.price is not None:
            payload["price"] = action.price
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CraigslistShopObservation]:
        obs_data = payload.get("observation", {})
        observation = CraigslistShopObservation(
            customer_message=obs_data.get("customer_message", ""),
            system_message=obs_data.get("system_message", ""),
            conversation_history=obs_data.get("conversation_history", []),
            item_category=obs_data.get("item_category", ""),
            item_title=obs_data.get("item_title", ""),
            item_description=obs_data.get("item_description", ""),
            listed_price=obs_data.get("listed_price", 0.0),
            current_offer_price=obs_data.get("current_offer_price"),
            turn=obs_data.get("turn", 0),
            episode_id=obs_data.get("episode_id", ""),
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
