# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional


@dataclass
class Arg:
    """Represents a single argument with a name and an optional weight"""
    name: str
    weight: Optional[float] = None

    def __str__(self):
        """Returns the string representation of the argument"""
        st = ""
        st += self.name
        if self.weight is not None:
            st += "("+str(self.weight)+")"
        return st

    def __eq__(self, other):
        """Checks equality based on the argument name"""
        if not isinstance(other, Arg):
            return False
        return self.name == other.name

    def __hash__(self):
        """Returns the hash based on the argument name"""
        return self.name  # hash((self.name, self.id))
