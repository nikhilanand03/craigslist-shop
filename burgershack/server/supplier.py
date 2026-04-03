# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stochastic supplier market engine for BurgerShack.

Simulates two suppliers with different price/reliability profiles,
seeded random-walk pricing, and supply shocks.
"""

import numpy as np

SUPPLIERS = {
    "cheapo_meats": {
        "base_prices": {
            "beef_patty": 1.00,
            "bun": 0.30,
            "lettuce": 0.20,
            "tomato": 0.25,
            "cheese": 0.35,
        },
        "reliability": 0.85,
    },
    "quality_farms": {
        "base_prices": {
            "beef_patty": 1.40,
            "bun": 0.45,
            "lettuce": 0.30,
            "tomato": 0.35,
            "cheese": 0.50,
        },
        "reliability": 0.99,
    },
}

INGREDIENTS = ["beef_patty", "bun", "lettuce", "tomato", "cheese"]


class SupplierEngine:
    """Manages supplier pricing with random walks and supply shocks."""

    def __init__(self, rng: np.random.Generator, volatility: float = 0.05,
                 shock_probability: float = 0.10):
        self.rng = rng
        self.volatility = volatility
        self.shock_probability = shock_probability

        # Initialize current prices from base prices
        self.current_prices: dict[str, dict[str, float]] = {}
        for supplier, info in SUPPLIERS.items():
            self.current_prices[supplier] = dict(info["base_prices"])

        # Pending deliveries: list of (delivery_turn, supplier, order_dict)
        self.pending_deliveries: list[tuple[int, str, dict[str, int]]] = []

        # Track which ingredients are out of stock at suppliers
        self.out_of_stock: dict[str, set[str]] = {s: set() for s in SUPPLIERS}

    def tick(self) -> None:
        """Advance prices by one step using a random walk."""
        for supplier in self.current_prices:
            for ingredient in self.current_prices[supplier]:
                if ingredient not in self.out_of_stock[supplier]:
                    noise = 1.0 + self.rng.normal(0, self.volatility)
                    self.current_prices[supplier][ingredient] = round(
                        self.current_prices[supplier][ingredient] * noise, 2
                    )
                    # Floor at 10% of base price
                    base = SUPPLIERS[supplier]["base_prices"][ingredient]
                    self.current_prices[supplier][ingredient] = max(
                        self.current_prices[supplier][ingredient],
                        round(base * 0.1, 2),
                    )

    def maybe_supply_shock(self) -> str | None:
        """Roll for a supply shock. Returns description if one occurs."""
        if self.rng.random() < self.shock_probability:
            ingredient = self.rng.choice(INGREDIENTS)
            shock_type = self.rng.choice(["price_spike", "out_of_stock"])

            if shock_type == "price_spike":
                for supplier in self.current_prices:
                    self.current_prices[supplier][ingredient] = round(
                        self.current_prices[supplier][ingredient] * 2.0, 2
                    )
                return f"Supply shock: {ingredient} prices doubled across all suppliers!"
            else:
                # One random supplier runs out of this ingredient
                supplier = self.rng.choice(list(SUPPLIERS.keys()))
                self.out_of_stock[supplier].add(ingredient)
                return (
                    f"Supply shock: {supplier} is out of stock on {ingredient}!"
                )
        return None

    def get_prices(self) -> dict[str, dict[str, float]]:
        """Return current prices, excluding out-of-stock items."""
        result = {}
        for supplier, prices in self.current_prices.items():
            result[supplier] = {
                ing: price
                for ing, price in prices.items()
                if ing not in self.out_of_stock[supplier]
            }
        return result

    def place_order(
        self, supplier: str, order: dict[str, int], current_turn: int
    ) -> tuple[float, str]:
        """
        Place a restock order. Returns (cost, status_message).
        Delivery arrives after 1 turn delay.
        """
        if supplier not in SUPPLIERS:
            return 0.0, f"Unknown supplier: {supplier}"

        reliability = SUPPLIERS[supplier]["reliability"]
        total_cost = 0.0
        items_ordered = {}

        for ingredient, qty in order.items():
            if ingredient not in INGREDIENTS:
                continue
            if ingredient in self.out_of_stock[supplier]:
                continue

            price = self.current_prices[supplier].get(ingredient, 0)
            total_cost += price * qty
            items_ordered[ingredient] = qty

        if not items_ordered:
            return 0.0, "No valid items in order."

        # Reliability check — order might fail
        if self.rng.random() > reliability:
            return total_cost, (
                f"Order from {supplier} failed to deliver! "
                f"You were still charged ${total_cost:.2f}."
            )

        # Schedule delivery for next turn
        self.pending_deliveries.append(
            (current_turn + 1, supplier, items_ordered)
        )

        total_cost = round(total_cost, 2)
        return total_cost, (
            f"Order placed with {supplier} for ${total_cost:.2f}. "
            f"Delivery arrives next turn."
        )

    def collect_deliveries(
        self, current_turn: int
    ) -> list[tuple[str, dict[str, int]]]:
        """Collect any deliveries that have arrived this turn."""
        arrived = []
        remaining = []
        for delivery_turn, supplier, items in self.pending_deliveries:
            if delivery_turn <= current_turn:
                arrived.append((supplier, items))
            else:
                remaining.append((delivery_turn, supplier, items))
        self.pending_deliveries = remaining
        return arrived
