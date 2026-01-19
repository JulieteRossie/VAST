# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


@dataclass
class Att:
    """Represents a directed attack between two arguments"""
    attacker_id: str
    attacked_id: str
    weight: Optional[int] = None

    def __eq__(self, other):
        """Checks equality based on attacker and attacked IDs"""
        return self.attacker_id == other.attacker_id and self.attacked_id == other.attacked_id

    def __str__(self):
        """Returns the string representation of the attack"""
        return str(self.attacker_id) + " --> " + str(self.attacked_id)

    def to_tuple(self):
        """Returns the attack as a tuple (attacker, attacked)"""
        return tuple((self.attacker_id, self.attacked_id))
