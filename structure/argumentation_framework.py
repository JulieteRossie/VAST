# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.y

from typing import List, Optional, Dict
import random
import warnings

from project_config import *
from structure.arguments import Args
from structure.argument import Arg
from structure.attack import Att
from structure.attacks import Attacks
from structure.graph import Graph
from structure.extension import Extension
from solver.asp import ASP

warnings.filterwarnings("ignore")


class AF:
    """Represents an Abstract Argumentation Framework (AF)"""
    arguments: Optional[Args] = None
    attacks: Optional[Attacks] = None
    graph: Optional[Graph] = None
    _solution: Optional[Dict[str, Dict[str, List[Extension]]]] = None
    _solver: Optional[ASP] = None
    # temporary mapping from old argument names to new formatted names
    new_to_old_mapping = None

    def __init__(self):
        self.arguments = Args()
        self.attacks = Attacks()
        self.reset_solution()
        self.defenders = None
        self.attackers = None

    def __str__(self):
        """Returns the APX representation of the AF"""
        strg = ""
        for arg in self.arguments:
            strg += "arg(" + arg + ").\n"
        if self.attacks.attacks is not None or self.attacks.attacks == []:
            for att in self.attacks:
                strg += "att(" + att.attacker_id + "," + att.attacked_id + ").\n"
        return strg

    def reset_solution(self):
        """Clears the computed extensions"""
        self._solution = {"extensions": dict()}

    def get_attackers(self):
        """Returns a dictionary mapping arguments to their attackers"""
        if self.attackers is None:
            self.attackers = dict()
            for arg in self.arguments:
                self.attackers[arg] = self.get_arg_attackers(arg)
        return self.attackers

    def get_arg_attackers(self, arg):
        """Returns the set of attackers for a specific argument"""
        attackers = set()

        for attack in self.attacks:
            if attack.attacked_id == arg:
                attackers.add(attack.attacker_id)
        return attackers

    def get_defenders(self):
        """Returns a dictionary mapping arguments to their defenders"""
        if self.defenders is None:
            self.defenders = dict()
            for arg in self.arguments:
                self.defenders[arg] = self.get_arg_defenders(arg)
        return self.defenders

    def get_arg_defenders(self, arg):
        """Return all arguments that defend a"""
        defenders = set()
        attackers = self.get_attackers()[arg]

        for attack in self.attacks:
            if attack.attacked_id in attackers:
                defenders.add(attack.attacker_id)

        return defenders

    def populate_random(self, n, treshold=0.75):
        """Generates a random AF with n arguments and attack probability"""
        for i in range(n):
            self.add_argument(Arg("a"+str(i+1)))
        for a in self.arguments:
            for b in self.arguments:
                if random.random() > treshold:
                    self.add_attack(Att(a, b))
        self.graph = Graph(edges=self.attacks.to_tuple(), nodes=list(self.arguments.keys()))

    def populate(self, filename=None, path=FrameworksPath, wrong_arg_name_format=False, text=None, isfile=True):
        """Parses APX format from file or string"""
        self.arguments = Args()
        self.attacks = Attacks()
        self._solution = {"extensions": dict()}
        if isfile and filename:
            file = open(path / filename, 'r')
            lines = file.readlines()
        elif text:
            lines = text.split("\n")
        else:
            raise ValueError("No apx was provided.")
        for line in lines:
            if line != '':
                tmp = line.split("(")[1].split(")")[0]
                if 'arg(' in line:
                    self.add_argument(Arg(tmp))
                elif 'att(' in line:
                    a1, a2 = tmp.split(",")
                    assert a1 in self.arguments.keys()
                    self.add_attack(Att(a1, a2))
        if wrong_arg_name_format:
            self.correct_name_format()
        self.graph = Graph(edges=self.attacks.to_tuple(), nodes=list(self.arguments.keys()))

    def correct_name_format(self):
        """Renames arguments to a standardized 'a1', 'a2'... format, useful for the ASP solver"""
        new_to_old_mapping = {}
        old_to_new_mapping = {}
        tmp_arguments = Args()

        for index, old_name in enumerate(self.arguments.keys(), start=1):
            new_name = f"a{index}"
            tmp_arguments.add(Arg(new_name))
            new_to_old_mapping[new_name] = old_name
            old_to_new_mapping[old_name] = new_name

        self.arguments = tmp_arguments

        new_attacks = Attacks()
        if self.attacks is None or len(self.attacks) == 0:
            new_attacks = Attacks()
        else:
            for attack in self.attacks:
                attacker_id = old_to_new_mapping.get(attack.attacker_id)
                attacked_id = old_to_new_mapping.get(attack.attacked_id)
                new_attacks.add(Att(attacker_id, attacked_id))
        self.new_to_old_mapping = new_to_old_mapping

        self.attacks = new_attacks

    def add_argument(self, element: Arg):
        self.arguments.add(element)

    def add_attack(self, element: Att):
        self.attacks.add(element)

    def distance(self, other: 'AF') -> int:
        """Computes the attack set distance (symmetric difference) between two AFs."""
        assert (len(self.arguments) == len(other.arguments))
        dist = 0
        for att in other.attacks:
            if att not in self.attacks:
                dist += 1
        for att in self.attacks:
            if att not in other.attacks:
                dist += 1
        return dist

    def draw_graph(self, ax, values=None, undirected_edges=[]):
        self.graph.draw_graph(ax, values, undirected_edges)

    def draw(self, votes=True, values=None, undirected_edges=[]):
        args = list(self.arguments.keys())
        self.graph.draw(values, undirected_edges, None, args)

    def get_extensions(self, semantic=None):
        """Retrieves extensions for the specified semantics"""
        if semantic is None:
            semantics = ["pref", "comp", "stab"]
        else:
            semantics = [semantic]
        for s in semantics:
            self.solve(s)
        if semantic is None:
            return self._solution["extensions"]
        return self._solution["extensions"][semantic]

    def solve(self, semantic):
        """Computes extensions via ASP solver"""
        if self._solver is None:
            self._solver = ASP()
        if len(self._solution["extensions"]) != 0 and semantic in self._solution["extensions"]:
            return
        match semantic:
            case "pref":  # preferred
                self._solution["extensions"]["pref"] = self._solver.solve_pref(self)
            case "comp":  # complete
                self._solution["extensions"]["comp"] = self._solver.solve_comp(self)
            case "stab":  # stable
                self._solution["extensions"]["stab"] = self._solver.solve_stab(self)
            case "cf":  # conflict free
                self._solution["extensions"]["cf"] = self._solver.solve_cf(self)
            case "adm":  # admissible
                self._solution["extensions"]["adm"] = self._solver.solve_adm(self)
            case "grou":  # admissible
                self._solution["extensions"]["grou"] = self._solver.solve_grou(self)

    def random_af(self, generation_type, **kwargs):
        """Generates a random AF graph structure"""
        self.graph = Graph([], [])
        self.graph.generate_random(generation_type, **kwargs)
        for arg in self.graph.graph.nodes:
            self.add_argument(Arg(name=arg))
        for att in self.graph.graph.edges:
            self.add_attack(Att(attacker_id=att[0], attacked_id=att[1]))
        self.correct_name_format()
        self.graph = Graph(edges=self.attacks.to_tuple(), nodes=list(self.arguments.keys()))
