# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

from cos.cos import COS
from structure.opinion_based_af import OBAF
from structure.extension import Extension


class MR(COS):
    """
    Majority Rule (MR).
    Aggregates opinions based on direct voter counts without using the argumentation framwork
    """
    def __init__(self, graph: Union[str, OBAF], semantic: str):
        super().__init__(graph, semantic)
        self.in_out_indec_map = None
        self.fill_in_out_undec()

    def majority_rule(self, arg):
        """Simple Majority Rule: 1 if IN > OUT, -1 if OUT > IN."""
        if self.in_out_indec_map[arg]['in'] > self.in_out_indec_map[arg]['out']:
            return 1
        elif self.in_out_indec_map[arg]['in'] < self.in_out_indec_map[arg]['out']:
            return -1
        else:
            return 0

    def fill_in_out_undec(self):
        """Pre-computes vote counts (IN, OUT, UNDEC) for every argument."""
        def m(argument):
            in_num = 0
            out_num = 0
            undec_num = 0
            for v in self.obaf.votes:
                if v[argument] == 1:
                    in_num += 1
                elif v[argument] == -1:
                    out_num += 1
                else:
                    undec_num += 1
            return in_num, out_num, undec_num

        if self.in_out_indec_map is None:
            self.in_out_indec_map = dict()
            for arg in self.obaf.af.arguments:
                i, o, u = m(arg)
                self.in_out_indec_map[arg] = {'in': i, 'out': o, 'undec': u}

    def print_everything(self):
        self.solve()
        print("Majority rule: arguments ", self.resulting_extensions["noparam"]["MR"])

    def solve(self, get_stats: bool = False):
        """Computes extensions for configured aggregation functions"""
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()

        functions = {
            "MR": self.majority_rule,
        }
        for method, function in functions.items():
            res = list()
            for arg in self.obaf.af.arguments:
                if function(arg) == 1:
                    res.append(arg)

            self.resulting_extensions["noparam"][method] = [Extension(res)]
