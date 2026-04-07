"""
Reward function for the negotiation environment.

Each episode is a single customer negotiation over a Craigslist-style item.

- Sold:     reward = sale_price / listed_price  ∈ [0, 1+]
            1.0 = sold at full listed price (perfect outcome)
            0.5 = sold at half the listed price
            >1  = sold above listed price (unlikely but possible)
- Walkaway: reward = 0.0 (no sale, no revenue)
"""


def compute_reward(
    sale_price: float,
    listed_price: float,
    served: bool,
) -> tuple[float, dict]:
    """
    Compute the reward for a completed negotiation.

    Args:
        sale_price: The price the item was sold at (0 if walkaway)
        listed_price: The original listed asking price
        served: Whether the sale was completed

    Returns:
        (reward, info_dict)
    """
    if served and listed_price > 0:
        reward = sale_price / listed_price
    elif served:
        reward = 1.0  # Edge case: listed_price is 0 or negative
    else:
        reward = 0.0

    info = {
        "sale_price": round(sale_price, 2) if served else 0.0,
        "listed_price": round(listed_price, 2),
        "price_retention": round(reward, 4) if served else 0.0,
        "served": served,
        "reward": round(reward, 4),
    }
    return round(reward, 4), info
