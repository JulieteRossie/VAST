# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from structure.opinion_based_af import OBAF
from typing import Optional, List, Dict, Union


class COS(ABC):
    """
    Abstract Base Class for a general Collective Opinion Semantics or COS.

    To add a new COS class, it should inherit from COS and implement all its abstract methods to be usable
    """
    def __init__(self, graph: Union[str, OBAF], semantic: str):
        """
        Initializes the COS wrapper.

        Args:
            graph: Either the .apx content in a string or an existing OBAF object.
            semantic: The underlying argumentation semantic to use (e.g., 'pref', 'comp').
        """
        self.obaf: Optional[OBAF] = None
        self.resulting_extensions: Optional[Dict[str, Dict]] = None
        self.resulting_statistics: Optional[Dict] = None
        if type(graph) is str:
            self.obaf = OBAF()
            self.obaf.populate(graph)
        else:
            self.obaf = graph
        self.semantic: str = semantic

    def __str__(self):
        return str(self.obaf) + "\n"

    @abstractmethod
    def print_everything(self):
        """Prints full details of the COS. Useful to compare the different COS and understand their operators"""
        ...

    @abstractmethod
    def solve(self):
        """Fills the self.resulting_extensions dictionary with the solution of the COS"""
        ...

    def repopulate_votes(self, filename, path, num_votes=None):
        """Wrapper to reload votes into the underlying OBAF."""
        self.obaf.repopulate_vote_files(filename, path, num_votes)

    def get_resulting_extensions(self) -> Dict:
        """Solves and returns the extensions of the COS."""
        if self.resulting_extensions is None:
            self.solve()
        return self.resulting_extensions
