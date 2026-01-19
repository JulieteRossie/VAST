# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional, Dict, List

from structure.argument import Arg


class Args:
    """Represents a collection of arguments behaving like a dictionary"""
    arguments: Optional[Dict[str, Arg]] = None
    __id_counter__: int = 0

    def __init__(self, value: Union[Arg, List[Arg], None] = None):
        """Initializes the collection with a single argument, a list of arguments, or empty"""
        match value:
            case Arg():
                self.add(value)
            case list():
                self.add_all(value)

    def __getitem__(self, key):
        return self.arguments[key]

    def __setitem__(self, key, value):
        self.arguments[key] = value

    def __delitem__(self, key):
        del self.arguments[key]

    def __iter__(self):
        return iter(self.arguments)

    def __len__(self):
        return len(self.arguments)

    def values(self):
        """Returns the list of argument objects"""
        return self.arguments.values()

    def items(self):
        """Returns the items (name, argument) of the collection"""
        return self.arguments.items()

    def keys(self):
        """Returns the argument names"""
        return self.arguments.keys()

    def __str__(self):
        """Returns the string representation of the argument set"""
        if self.arguments is None:
            return "empty.apx arguments list"
        st = "{"
        for a in list(self.arguments.values())[:-1]:
            st += str(a)+", "
        st += str(list(self.arguments.values())[-1])
        st += "}"
        return st

    def add_all(self, elements):
        """Adds a list of arguments to the collection"""
        for e in elements:
            self.add(e)

    def add(self, element):
        """Adds a single argument if its name is unique"""
        if self.arguments is None:
            self.arguments = dict()
        if element.name not in self.arguments:
            self.arguments[element.name] = element
        else:
            print("ARGS::add error: new argument will not be added to the list, it's name is not unique")
            # warnings.warn("ARGS::add error: new argument will not be added to the list, it's name is not unique")

    def pop(self, a):
        """Removes and returns the argument with the given name"""
        self.arguments.pop(a)
