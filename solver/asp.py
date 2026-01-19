# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clingo
from typing import List

from solver.afsolver import AFSolver
from structure.extension import Extension


class ASP(AFSolver):
    """Solver implementation using Clingo (ASP) to compute extensions"""
    def __init__(self):
        super().__init__()

    def solve(self, af, n, semantics):
        """Executes the ASP solver with the given AF and semantics encoding"""
        asp = str(af) + semantics + "#show in/1."

        control = clingo.Control(message_limit=0) # added message limit 0 to avoid messages when there are no attacks
        control.add("base", [], asp)
        control.ground([("base", [])])

        control.configuration.solve.models = n

        models = []

        def on_model(model):
            nonlocal models
            m = []
            for atom in model.symbols(shown=True):
                # in(a3) -> 3
                m.append(str(atom.arguments[0]))

            models.append(m)

        answer = control.solve(on_model=on_model)
        return models

    def solve_adm(self, af):
        """Given an AF, returns n admissible extensions (if n=0, returns all)"""
        n = 0
        semantics = """
        in(X) :- not out(X), arg(X).
        out(X) :- not in(X), arg(X).
    
        :- in(X), in(Y), att(X,Y).
    
        defeated(X) :- in(Y), att(Y,X).
    
        not_defended(X) :- att(Y,X), not defeated(Y).
    
        :- in(X), not_defended(X)."""

        exts = self.solve(af, n, semantics)
        exts_tuples: List[Extension] = [Extension(s) for s in exts]
        return exts_tuples

    def solve_pref(self, af):
        """Given an AF, returns n preferred extensions (if n=0, returns all)"""
        n = 0
        semantics = """
        in(X) :- not out(X), arg(X).
        out(X) :- not in(X), arg(X).
    
        :- in(X), in(Y), att(X,Y).
    
        defeated(X) :- in(Y), att(Y,X).
        not_defended(X) :- att(Y,X), not defeated(Y).
    
        :- in(X), not_defended(X).
    
        not_trivial :- out(X).
        ecl(X) : out(X) :- not_trivial.
        spoil | ecl(Z) : att(Z,Y) :- ecl(X), att(Y,X).
        spoil :- ecl(X), ecl(Y), att(X,Y).
        spoil :- in(X), ecl(Y), att(X,Y).
        ecl(X) :- spoil, arg(X).
        :- not spoil, not_trivial.
        """

        exts = self.solve(af, n, semantics)
        exts_tuples: List[Extension] = [Extension(s) for s in exts]
        return exts_tuples

    def solve_comp(self, af):
        """Given an AF, returns n complete extensions (if n=0, returns all)"""
        n = 0
        semantics = """
        in(X) :- not out(X), arg(X).
        out(X) :- not in(X), arg(X).
    
        :- in(X), in(Y), att(X,Y).
    
        defeated(X) :- in(Y), att(Y,X).
    
        not_defended(X) :- att(Y,X), not defeated(Y).
    
        :- in(X), not_defended(X).
    
        :- out(X), not not_defended(X)."""

        exts = self.solve(af, n, semantics)
        exts_tuples: List[Extension] = [Extension(s) for s in exts]
        return exts_tuples

    def solve_cf(self, af):
        """Given an AF, returns n admissible extensions (if n=0, returns all) """
        n = 0
        semantics = """
        in(X) :- not out(X), arg(X).
        out(X) :- not in(X), arg(X).
    
        :- in(X), in(Y), att(X,Y)."""

        exts = self.solve(af, n, semantics)
        exts_tuples: List[Extension] = [Extension(s) for s in exts]
        return exts_tuples

    def solve_stab(self, af):
        """Given an AF, returns n stable extensions (if n=0, returns all)"""
        n = 0
        semantics = """
        in(X) :- not out(X), arg(X).
        out(X) :- not in(X), arg(X).

        :- in(X), in(Y), att(X,Y).

        defeated(X) :- in(Y), att(Y,X).

        :- out(X), not defeated(X).
        """

        exts = self.solve(af, n, semantics)
        exts_tuples: List[Extension] = [Extension(s) for s in exts]
        return exts_tuples

    def solve_grou(self, af):
        """Given an AF, returns the grounded extension"""
        complete_exts = self.solve_comp(af)

        def find_min_sets(extensions):
            non_min_sets = []
            for ext in extensions:
                for other_ext in extensions:
                    if other_ext.issubset(ext) and ext != other_ext:
                        non_min_sets.append(ext)
            answer = [item for item in extensions if item not in non_min_sets]
            return answer

        min_exts = find_min_sets(complete_exts)

        return min_exts
