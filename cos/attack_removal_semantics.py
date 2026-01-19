# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Dict
from structure.opinion_based_af import OBAF

from cos.helpers import PAF
from cos.cos import COS


class ARS(COS):
    """
    Attack Removal Semantics (ARS).
    Computes the initial scores for each argument based on the votes using the initial function from SAF paper by Leite and Martins,
    then uses a Preference-Based AF (PAF) by Amgoud and Cayrol to opbtain extensions by removing attacks where the attacker
     is weaker than the target.
    """
    initial: Dict[str, float]
    initial_abs: Dict[str, float]
    final: Dict[str, int]
    semantic: str

    def __init__(self, graph: Union[str, OBAF], semantic):
        super().__init__(graph, semantic)
        self.initial = self.__get_initial__()
        self.initial_abs = self.__get_initial__(True)
        self.final = None

    def __get_votes_dict__(self, w_wabs=False, zero_value=0.5):
        """Aggregates raw vote counts into positive, negative, and abstentions for each of the arguments"""
        if not self.obaf.votes:
            return
        res = dict()
        for argument, _ in self.obaf.af.arguments.items():
            res[argument] = [0, 0, 0, 0, 0, 0]
            for vote in self.obaf.votes:
                if int(vote[argument]) > 0:
                    res[argument][0] += int(vote[argument])
                    res[argument][1] += 1
                if w_wabs and int(vote[argument]) == 0:
                    res[argument][2] += zero_value
                    res[argument][3] += 1
                if int(vote[argument]) < 0:
                    res[argument][4] += abs(int(vote[argument]))
                    res[argument][5] += 1
        return res

    def __get_initial__(self, get_zeros=False, epsilon=0.001):
        """Computes initial argument scores based on the ratio of support vs rejection"""
        if not self.obaf.votes:
            return

        votes_dict = self.__get_votes_dict__(get_zeros)
        res = dict()
        if not get_zeros:
            for key, vote in votes_dict.items():
                res[key] = vote[0] / (vote[1] + vote[5] + epsilon)
        else:
            for key, vote in votes_dict.items():
                res[key] = vote[0] + vote[2] / (vote[1] + vote[3] + vote[5] + epsilon)
        return res

    def get_extensions(self, w_abs=False, grounded=False):
        """Uses PAF helper to filter attacks based on scores and re-evaluate semantics"""
        if w_abs:
            pref = PAF(self.obaf, self.obaf.af.get_extensions(self.semantic), self.initial_abs)
        else:
            pref = PAF(self.obaf, self.obaf.af.get_extensions(self.semantic), self.initial)
        return pref.reevaluate_graph(semantic=self.semantic)

    def print_rounded(self, tab, to_round=2):
        """Print helper"""
        print("{", end="")
        for i, v in dict(sorted(tab, reverse=True, key=lambda key_val: key_val[1])).items():
            print(i, ":", round(v, to_round), end=", ")
        print("}")

    def solve(self):
        """Computes the extensions after attack removal."""
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"]["no-abs"] = list(self.get_extensions(False))
        # self.resulting_extensions["noparam"]["w-abs"] = list(self.get_extensions(False, w_abs=True))


    def print_everything(self):
        if not self.obaf.votes:
            print("no votes")
            return
        print("SAF initial rankings:", end="")
        self.print_rounded(self.initial.items())
        print("ARS", '\033[43m ' + ",".join([str(x) for x in self.get_extensions()]) + '\033[49m ')
