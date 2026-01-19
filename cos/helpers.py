# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Dict
from copy import deepcopy
from itertools import combinations, chain
from dataclasses import dataclass
import numpy as np

from structure.opinion_based_af import OBAF
from structure.attacks import Attacks
from structure.arguments import Args
from structure.extension import Extension
from structure.labelling import Labelling


def ext_to_labeling(extention: Extension, arguments: Args, attacks: Attacks) -> Labelling:
    """Converts an extension to a 3-valued labeling (IN, OUT, UNDEC)."""
    lab = {'in': [x for x in extention], "out": [], "undec": []}
    tmp = deepcopy(arguments)
    for a in lab['in']:
        tmp.pop(a)
    for a in tmp:
        for att in attacks:
            if a == att.attacked_id and att.attacker_id in lab['in'] and a not in lab['out']:
                lab['out'].append(a)
    for a in lab['out']:
        tmp.pop(a)
    if len(tmp) > 0:
        lab['undec'] = [x for x in tmp.keys()]
    return Labelling(lab["in"], lab["out"], lab["undec"])


def ext_to_vec(ext, arguments):
    """Converts an extension to a numerical vector: 1 if in extension, -1 otherwise."""
    vec = dict()
    for a in arguments:
        if a in ext:
            vec[a] = 1
        else:
            vec[a] = -1
    return vec


@dataclass
class PAF:
    """
    Preference-based Argumentation Framework helper.
    Evaluates arguments based on scores and filters attacks where the attacker is weaker than the target.
    """
    obaf: Optional[OBAF] = None
    semantic_extensions_vecs: Optional[List[Dict[str, int]]] = None
    scores: Optional[Dict[str, float]] = None

    def __post_init__(self):
        self.obaf = deepcopy(self.obaf)

    def reset_attacks(self) -> float:
        new_att = Attacks()
        for att in self.obaf.af.attacks:
            if self.scores[att.attacker_id] >= self.scores[att.attacked_id]:
                new_att.add(att)
        difference = len(new_att) * 100 / len(self.obaf.af.attacks)
        self.obaf.af.attacks = new_att
        return difference

    def reevaluate_graph(self, semantic):
        self.obaf.af.reset_solution()
        self.reset_attacks()
        self.obaf.af.solve(semantic)
        return self.obaf.af.get_extensions(semantic)
