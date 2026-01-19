# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List

from structure.extension import Extension


class AFSolver(ABC):
    """
    Abstract base class for Argumentation Framework solvers
    If you would like to change the ASP solver, the new solver should inherit from this class
    """
    def __init__(self):
        pass

    @abstractmethod
    def solve_cf(self, af) -> List[Extension]:
        """Returns conflict-free extensions."""
        ...

    @abstractmethod
    def solve_adm(self, af) -> List[Extension]:
        """Returns admissible extensions."""
        ...

    @abstractmethod
    def solve_comp(self, af) -> List[Extension]:
        """Returns complete extensions."""
        ...

    @abstractmethod
    def solve_pref(self, af) -> List[Extension]:
        """Returns preferred extensions."""
        ...

    def solve_stab(self, af) -> List[Extension]:
        """Returns stable extensions."""
        ...

    def solve_semi_stab(self, af) -> List[Extension]:
        """Returns semi stable extensions."""
        raise NotImplementedError

    def solve_stage(self, af) -> List[Extension]:
        """Returns stage extensions."""
        raise NotImplementedError
