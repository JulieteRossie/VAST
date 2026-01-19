# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List

from structure.arguments import Args


class Extension:
    """Represents a set of arguments (e.g., a solution to an argumentation framework)"""
    def __init__(self, arguments: Union[str, Args, List[str], List[int]]):
        self.arguments = frozenset(arguments)

    def __str__(self):
        """Returns the string representation of the extension"""
        return "{" + ",".join([str(x) for x in self.arguments]) + "}"

    def __len__(self):
        return len(self.arguments)

    def __eq__(self, other):
        if isinstance(other, Extension):
            return self.arguments == other.arguments
        if isinstance(other, str):
            res = True
            for arg in other:
                if arg not in self.arguments:
                    res = False
            return res

    def __iter__(self):
        return iter(self.arguments)

    def __hash__(self):
        return hash(self.arguments)

    def __contains__(self, item):
        return item in self.arguments

    def __repr__(self):
        return f"{self.__str__()}"

    def intersection(self, other):
        """Returns a new Extension containing the intersection with another object (a string or an another extension)"""
        res = []
        if type(other) is str:  # String
            for arg in other:
                if arg in self.arguments:
                    res.append(arg)
        if type(other) is Extension:  # Extension
            for arg in other.arguments:
                if arg in self.arguments:
                    res.append(arg)
        else:  # Vote
            return other.intersection(self)
        return Extension(res)

    def union(self, other: "Extension"):
        """Returns a new Extension containing the union of arguments between the two extensions"""
        return Extension(list(self.arguments.union(other.arguments)))

    def get_args(self):
        """Returns the frozen set of arguments"""
        return self.arguments

    def hamming_distance(self, other: "Extension"):
        """Computes the Hamming distance between two extensions"""
        max_size = max(len(self.arguments), len(other.arguments))

        intersection_size = len(set(self.arguments).intersection(other.arguments))
        hamming_distance = max_size - intersection_size
        return hamming_distance

    def is_empty(self):
        """Checks if the extension contains no arguments"""
        return len(self.arguments) == 0

    def get_vector(self, arguments):
        """Transforms the extensions into a vector with 1 if the argument is in the extension and -1 otherwise"""
        return [1 if arg in self.arguments else -1 for arg in arguments]

    def issubset(self, other):
        """Checks if this extension is a subset of another"""
        if isinstance(other, Extension):
            return self.arguments.issubset(other.arguments)
        return False
