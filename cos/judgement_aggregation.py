# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union
from copy import deepcopy

from structure.opinion_based_af import OBAF
from structure.labelling import Labelling
from structure.extension import Extension
from cos.helpers import ext_to_labeling
from cos.cos import COS


class JA(COS):
    """
    Judgment Aggregation (JA).
    Implements Skeptical, Credulous, and Super-Credulous aggregation operators based on the work of Pigozzi and Caminiada.
    """
    initial: Labelling

    def __init__(self, graph: Union[str, OBAF], semantic: str):
        """solver_type is initialized to adm as it is used by all operators"""
        super().__init__(graph, semantic)
        self.solver_type = 'adm'

    def sio(self):
        """
        Skeptical Initial Labelling (sio):
        - A is labelled in if everyone agrees A is in.
        - A is labelled out if everyone agrees A is out.
        - A is labelled undec in all other cases
        """
        res = {
            "in": [],
            "out": [],
            "undec": []
        }

        for arg in self.obaf.af.arguments.keys():
            in_l = 0
            out_l = 0
            for vote in self.obaf.votes:
                if int(vote[arg]) == -1:
                    out_l += 1
                if int(vote[arg]) == 1:
                    in_l += 1
            if in_l >= len(self.obaf.votes):
                res["in"].append(arg)
            elif out_l >= len(self.obaf.votes):
                res["out"].append(arg)
            else:
                res["undec"].append(arg)
        self.initial = Labelling(res["in"], res["out"], res["undec"])

    def up_complete(self, credulous: Labelling) -> List[Labelling]:
        """
        Expansion to Complete: Finds the 'smallest' (min In set) complete labellings
        that extend the credulous labelling.
        """
        extensions_semantics = self.obaf.af.get_extensions(self.solver_type)
        labellings_semantics = []
        for ext in extensions_semantics:
            labellings_semantics.append(ext_to_labeling(ext, self.obaf.af.arguments, self.obaf.af.attacks))

        if len(credulous.in_args) == 0 and len(credulous.out_args) == 0:
            return [Labelling([], [], [x for x in self.obaf.af.arguments])]

        in_score = 1000000
        shortest_labellings = []
        for l in labellings_semantics:
            if set(credulous.in_args).issubset(l.in_args) and set(credulous.out_args).issubset(l.out_args):
                if len(l.in_args) < in_score:
                    in_score = len(l.in_args)
                    shortest_labellings = [l]
                elif len(l.in_args) == in_score:
                    shortest_labellings.append(l)
        return shortest_labellings

    def down_admissible(self) -> List[Labelling]:
        """
        Contraction to Admissible: Finds the 'largest' (max In set) admissible labellings
        that are subsets of the initial labelling
        """
        extensions_semantics = self.obaf.af.get_extensions(self.solver_type)
        labellings_semantics = []
        for ext in extensions_semantics:
            labellings_semantics.append(ext_to_labeling(ext, self.obaf.af.arguments, self.obaf.af.attacks))

        if len(self.initial.in_args) == 0 and len(self.initial.out_args) == 0:
            return [Labelling([], [], [x for x in self.obaf.af.arguments])]

        in_score = 0
        longest_labellings = []
        for l in labellings_semantics:
            if set(l.in_args).issubset(self.initial.in_args) and set(l.out_args).issubset(self.initial.out_args):
                if len(l.in_args) > in_score:
                    in_score = len(l.in_args)
                    longest_labellings = [l]
                elif len(l.in_args) == in_score:
                    longest_labellings.append(l)
        return longest_labellings

    def skeptical_aggregation(self) -> List[Labelling]:
        """Computes skeptical outcome: Unanimous consensus (sio) -> Down-Admissible."""
        if not self.obaf.votes:
            return "no votes"
        
        self.sio()
        return self.down_admissible()

    def cio(self):
        """
        Credulous Initial Labelling (cio):
        - A is labelled in if someone thinks A is in and nobody thinks A is out.
        - A is labelled out if someone thinks A is out and nobody thinks is in.
        - A is labelled undec in all other cases.
        """
        res = {
            "in": [],
            "out": [],
            "undec": []
        }

        for arg in self.obaf.af.arguments.keys():
            in_l = 0
            out_l = 0
            for vote in self.obaf.votes:
                if int(vote[arg]) <= 0:
                    out_l += 1
                if int(vote[arg]) >= 0:
                    in_l += 1
            if in_l >= len(self.obaf.votes):
                res["in"].append(arg)
            elif out_l >= len(self.obaf.votes):
                res["out"].append(arg)
            else:
                res["undec"].append(arg)
        self.initial = Labelling(res["in"], res["out"], res["undec"])

    def credulous_aggregation(self) -> List[Labelling]:
        """Computes credulous outcome: Non-conflicting consensus (cio) -> Down-Admissible."""
        if not self.obaf.votes:
            return "no votes"

        self.cio()
        return self.down_admissible()

    @staticmethod
    def illegally_undec(lab: Labelling, arg, arguments, attacks):
        for a in arguments:
            total = 0
            cpt = 0
            for att in attacks:
                if a == att.attacker_id and arg == att.attacked_id:
                    if a in lab.in_args:
                        return True, 'out'
                    if a in lab.out_args:
                        total += 1
                    cpt += 1
            if total == cpt:
                return True, 'in'
        return False, 'undec'

    def sco(self) -> List[Labelling]:
        """Expands the credulous result to the closest valid complete labelling"""
        credulous = self.credulous_aggregation()
        if len(credulous) == 1:
            return self.up_complete(credulous[0])
        else:
            res = []
            labs = deepcopy(credulous)
            for lab in labs:
                tmp = self.up_complete(lab)
                if type(tmp) == list:
                    for ext in tmp:
                        res.append(ext)
                else:
                    res.append(tmp)
            return res

    def super_credulous_aggregation(self) -> List[Labelling]:
        """Calculates super-credulous outcome: Expands the credulous result to the closest valid complete labelling."""
        if not self.obaf.votes:
            return "no votes"

        return self.sco()

    def solve(self):
        """Computes all JA variants: skeptical, credulous, and super-credulous, and stores them in the resulting_extensions dict"""
        self.resulting_extensions = dict()
        self.resulting_extensions["noparam"] = dict()
        self.resulting_extensions["noparam"]["skeptical"] = [Extension(x.in_args) for x in self.skeptical_aggregation()]
        self.resulting_extensions["noparam"]["credulous"] = [Extension(x.in_args)for x in self.credulous_aggregation()]
        self.resulting_extensions["noparam"]["super-credulous"] = [Extension(x.in_args) for x in self.super_credulous_aggregation()]

    @staticmethod
    def print_color(labellings: List[Labelling]):
        res = "["
        for lab in labellings:
            in_args, out_args, undec_args = lab
            res += '{'
            res += "\'in\': " + ' \033[43m ' + ','.join(in_args) + ' \033[49m ' + ', '
            res += "\'out\': " + ','.join(out_args) + ', '
            res += "\'undec\': " + ','.join(undec_args)
            res += '}'
        res += ']'
        return res

    def print_everything(self):
        print("Skeptical: ", JA.print_color(self.skeptical_aggregation()))
        print("Credulous: ", JA.print_color(self.credulous_aggregation()))
        print("Super credulous: ", JA.print_color(self.super_credulous_aggregation()))
