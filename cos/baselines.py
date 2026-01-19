# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from random import choice
from typing import Union

from cos.cos import COS
from structure.opinion_based_af import OBAF
from structure.extension import Extension


class AllExtensionsBaseline(COS):
    """Returns all extensions found by the underlying semantics (pref, comp, etc.)"""
    def __init__(self, graph: Union[str, OBAF], semantic_ext):
        super().__init__(graph, semantic_ext)
        self.semantic = semantic_ext

    def print_everything(self):
        self.solve()
        for ext in self.resulting_extensions:
            print(ext)

    def solve(self):
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"][self.semantic] = self.obaf.af.get_extensions(self.semantic)


class AllArgumentsBaseline(COS):
    """Returns a single extension containing all arguments in the framework"""
    def __init__(self, graph: Union[str, OBAF], semantic_ext):
        super().__init__(graph, semantic_ext)

    def print_everything(self):
        self.solve()
        for ext in self.resulting_extensions:
            print(ext)

    def solve(self):
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"]["all"] = [Extension(self.obaf.af.arguments)]


class RandomExtensionBaseline(COS):
    """Returns exactly one random extension from the underlying semantics"""
    def __init__(self, graph: Union[str, OBAF], semantic_ext):
        super().__init__(graph, semantic_ext)
        self.semantic = semantic_ext

    def print_everything(self):
        self.solve()
        for ext in self.resulting_extensions:
            print(ext)

    def solve(self):
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"]["random"] = [choice(self.obaf.af.get_extensions(self.semantic))]


class EmptyExtensionBaseline(COS):
    """Returns a single empty extension."""
    def __init__(self, graph: Union[str, OBAF], semantic_ext):
        super().__init__(graph, semantic_ext)

    def print_everything(self):
        self.solve()
        for ext in self.resulting_extensions:
            print(ext)

    def solve(self):
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"]["empty"] = [Extension([""])]
