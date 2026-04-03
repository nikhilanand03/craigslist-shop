# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward function for BurgerShack.

Each episode is a single customer negotiation. The reward is based on
the delta between the sale price and the cost floor (ingredient cost + margin).

- Served: reward = (sale_price - cost_floor) / (menu_price - cost_floor)
  Normalized to [0, 1] when selling between cost_floor and menu_price,
  and > 1 if somehow selling above menu price.
- Walkaway: reward = -1.0 (worst outcome — no money at all)

This makes walkaway strictly worse than selling at cost_floor (reward=0),
which is strictly worse than selling at menu price (reward=1).
"""


def compute_reward(
    sale_price: float,
    menu_price: float,
    cost_floor: float,
    served: bool,
) -> tuple[float, dict]:
    """
    Compute the reward for a completed customer interaction.

    Args:
        sale_price: The price the burger was sold at (0 if walkaway)
        menu_price: The listed menu price
        cost_floor: Minimum acceptable price (ingredient cost + margin)
        served: Whether the customer was served

    Returns:
        (reward, info_dict)
    """
    price_range = menu_price - cost_floor if menu_price > cost_floor else 1.0

    if served:
        # Reward is the normalized profit above cost floor
        # 0.0 = sold at cost floor (broke even on margin)
        # 1.0 = sold at full menu price (maximum profit)
        # Negative if sold below cost floor (shouldn't happen but possible)
        reward = (sale_price - cost_floor) / price_range
    else:
        # Walkaway: worst outcome. The agent made nothing and lost the customer.
        reward = -1.0

    info = {
        "sale_price": round(sale_price, 2) if served else 0.0,
        "menu_price": round(menu_price, 2),
        "cost_floor": round(cost_floor, 2),
        "profit": round(sale_price - cost_floor, 2) if served else 0.0,
        "served": served,
        "reward": round(reward, 4),
    }
    return round(reward, 4), info
