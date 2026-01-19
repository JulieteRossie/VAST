# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Optional

from structure.attack import Att

class Attacks:
    """Represents a collection of attacks"""
    attacks: Optional[List[Att]] = None

    def __init__(self, value: Union[Att, List[Att], None] = None):
        """Initializes the collection with a single attack, a list of attacks, or empty"""
        match value:
            case Att():
                self.add(value)
            case list():
                self.add_all(value)

    def __str__(self):
        """Returns the string representation of the set of attacks"""
        if self.attacks is None:
            return "attacks list is empty.apx"
        st = "{"
        for a in self.attacks[:-1]:
            st += str(a)+", "
        st += str(self.attacks[-1])
        st += "}"
        return st

    def __getitem__(self, item):
        return self.attacks[item]

    def __len__(self):
        return len(self.attacks) if self.attacks else 0

    def remove(self, element):
        """Removes an attack from the list"""
        return self.attacks.remove(element)

    def add_all(self, elements):
        """Adds a list of attacks to the collection"""
        for e in elements:
            self.add(e)

    def add(self, element):
        """Adds a single attack if it is unique"""
        if self.attacks is None:
            self.attacks = list()
        if element not in self.attacks:
            self.attacks.append(element)
        else:
            print("ATTACKS::add error: new attack will not be added to the list, it's not unique")

    def find(self, other):
        """Checks if a specific attack exists in the collection"""
        for a in self.attacks:
            if a == other:
                return True
        return False

    def to_tuple(self):
        """Converts all attacks in the collection to a list of tuples"""
        if self.attacks:
            return [x.to_tuple() for x in self.attacks]
        else:
            return []
