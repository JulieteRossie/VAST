# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List

from structure.arguments import Args


class Labelling:
    """Represents a 3-valued labelling (in, out, undec) of arguments"""
    def __init__(self, in_args: Union[str, Args, List[str], List[int]],
                 out_args: Union[str, Args, List[str], List[int]],
                 undec_args: Union[str, Args, List[str], List[int]]):
        self.in_args = frozenset(in_args)
        self.out_args = frozenset(out_args)
        self.undec_args = frozenset(undec_args)

    def __str__(self):
        """Returns the string representation of the labelling"""
        return "{ in = " + str(self.in_args) + ", out = " + str(self.out_args) + ", undec = " + str(self.undec_args) + "}"

    def __eq__(self, other):
        """Checks equality based on the three sets"""
        if isinstance(other, Labelling):
            return self.in_args == other.in_args and self.out_args == other.out_args and self.undec_args == other.undec_args
        else:
            return False

    def __hash__(self):
        """Returns the hash based on the argument sets"""
        return hash((self.in_args, self.out_args, self.undec_args))

    def __iter__(self):
        """Makes the Labelling class iterable. Iterates over the 'in_args', 'out_args', and 'undec_args'."""
        return iter((self.in_args, self.out_args, self.undec_args))
