# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Episode state machine for BurgerShack.

Each episode is a single customer negotiation.
"""

from enum import Enum


class Phase(str, Enum):
    CUSTOMER_ARRIVES = "customer_arrives"
    NEGOTIATION = "negotiation"
    TRANSACTION = "transaction"
    WALKAWAY = "walkaway"


class StateMachine:
    """Tracks the current phase of a single-customer episode."""

    VALID_TRANSITIONS = {
        Phase.CUSTOMER_ARRIVES: {Phase.NEGOTIATION},
        Phase.NEGOTIATION: {Phase.NEGOTIATION, Phase.TRANSACTION, Phase.WALKAWAY},
    }

    def __init__(self):
        self.phase = Phase.CUSTOMER_ARRIVES

    def transition(self, target: Phase) -> None:
        valid = self.VALID_TRANSITIONS.get(self.phase, set())
        if target not in valid:
            raise ValueError(
                f"Invalid transition: {self.phase.value} -> {target.value}. "
                f"Valid targets: {[p.value for p in valid]}"
            )
        self.phase = target

    def can_transition(self, target: Phase) -> bool:
        valid = self.VALID_TRANSITIONS.get(self.phase, set())
        return target in valid
