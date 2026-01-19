# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from copy import deepcopy
from typing import Union, Dict, FrozenSet

import structure.extension
from structure.attacks import Attacks


class Vote:
    """Represents a voter's opinion (+1 yes, -1 no, 0 abstention) on arguments"""
    vote_dict: Dict[str, int]

    def __init__(self, name: str, yeses: FrozenSet = None, nos: FrozenSet = None, zeros: FrozenSet = None, dico= None):
        """Initializes vote from explicit sets or a dictionary"""
        self.vote_dict = dict()
        self.name = name
        if yeses is not None:
            for arg in yeses:
                self.add(arg, 1)
        if nos is not None:
            for o in nos:
                if o not in yeses:
                    self.add(o, -1)
        if zeros is not None:
            for o in zeros:
                if o not in yeses and o not in nos:
                    self.add(o, 0)
        elif dico is not None:
            self.vote_dict = dict()
            self.vote_dict = deepcopy(dico)

    def __getitem__(self, item):
        return self.vote_dict.__getitem__(item)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "Vote") -> bool:
        return self.name == other.name

    def __str__(self):
        res = self.name + ": ("
        for k, v in self.vote_dict.items():
            res += str(k) + ": " + str(v) + ", "
        res += ")"
        return res

    def add(self, arg_id: str, value: int, prints=False):
        """Adds the vote for argument arg_id if it does not already exist"""
        if arg_id in self.vote_dict and prints:
            print("VOTE::add error: new argument will not be added to the list, as it already exists")
        else:
            self.vote_dict[arg_id] = value

    def get_vector(self, arguments):
        """Returns the vote values corresponding to a list of arguments"""
        vector = []
        for arg in arguments:
            vector.append(self.vote_dict[arg])
        return vector
        # return [x for x in self.vote_dict.values()]

    def to_ext(self):
        """Converts the 'yes' votes to an Extension object"""
        return structure.extension.Extension(self.yes())

    def to_frozenset(self):
        """Returns a frozenset of arguments voted 'yes'"""
        return frozenset(self.yes())

    def intersection(self, other: Union["Vote", structure.extension.Extension, str]):
        """Computes intersection based on 'yes' votes or identical vote values"""
        res = set()
        if type(other) is structure.extension.Extension or type(other) is str:
            for arg in self.vote_dict.keys():
                if self.vote_dict[arg] >= 1 and arg in other:
                    res.add(arg)
        else:
            for arg in self.vote_dict.keys():
                if self.vote_dict[arg] == other.vote_dict[arg]:
                    res.add(arg)
        return res

    def intersection_abs(self, other: Union["Vote", structure.extension.Extension, str]):
        """Computes intersection treating abstentions as potential matches"""
        res = set()
        if type(other) is structure.extension.Extension or type(other) is str:
            for arg in self.vote_dict.keys():
                if self.vote_dict[arg] >= 0 and arg in other:
                    res.add(arg)
        else:
            for arg in self.vote_dict.keys():
                if self.vote_dict[arg] == other.vote_dict[arg] or other.vote_dict[arg] == 0 or self.vote_dict[arg] == 0:
                    res.add(arg)
        return res

    def is_consistent(self, attacks: Attacks):
        """Checks if the vote respects the conflict-free property (no two 'yes' arguments attack each other)"""
        for attack in attacks:
            if self.vote_dict[attack.attacker_id] == 1:
                if self.vote_dict[attack.attacked_id] == 1:
                    return False
        return True

    def get_consistent(self, attacks: Attacks):
        """Attempts to resolve inconsistencies by flipping conflicting votes to -1"""
        modified_vote = deepcopy(self)

        inconsistent_attacks = [
            attack for attack in attacks
            if modified_vote.vote_dict[attack.attacker_id] == 1 and
               modified_vote.vote_dict[attack.attacked_id] == 1
        ]

        for attack in inconsistent_attacks:
            r = random.choice([1, 2])
            if r == 1:
                modified_vote.vote_dict[attack.attacked_id] = -1
                if not modified_vote.is_consistent(attacks):
                    modified_vote.vote_dict[attack.attacker_id] = -1
            else:
                modified_vote.vote_dict[attack.attacker_id] = -1
                if not modified_vote.is_consistent(attacks):
                    modified_vote.vote_dict[attack.attacked_id] = -1

        if modified_vote.is_consistent(attacks):
                return modified_vote

        raise ValueError("Unable to make vote consistent with minimal changes")

    def yes(self):
        """Returns arguments voted 1 (Yes)"""
        res = []
        for arg, value in self.vote_dict.items():
            if value >= 1:
                res.append(arg)
        return res

    def no(self):
        """Returns arguments voted -1 (No)"""
        res = []
        for arg, value in self.vote_dict.items():
            if value <= -1:
                res.append(arg)
        return res

    def maybe(self):
        """Returns arguments voted 0 (Abstention)"""
        res = []
        for arg, value in self.vote_dict.items():
            if value == 0:
                res.append(arg)
        return res

    def write(self):
        """Formats the vote for file output"""
        res = "vote(" + self.name
        for y in self.yes():
            res += ",(" + str(y) + ',1)'
        for n in self.no():
            res += ",(" + str(n) + ',-1)'
        res += ")."
        return res
