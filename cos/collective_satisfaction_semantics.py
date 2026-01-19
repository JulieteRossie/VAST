# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

from structure.opinion_based_af import OBAF
from structure.extension import Extension
from structure.vote import Vote
from cos.helpers import ext_to_vec
from cos.cos import COS
from cos.attack_removal_semantics import ARS


class CSS(COS):
    """
    Collective Satisfaction Semantics (CSS).
    Evaluates the underlying extensions based on voter satisfaction, dissatisfaction, and utility. And the aggregates in
    either egalitarian (min and leximin) or utilitarian (sum) ways.
    """
    def __init__(self, graph: Union[str, OBAF], semantic: str):
        """
        The methods and aggregation are initiated to all possible methods and aggregation so that the resulting_extensions
        dict would be filled with all solutions on the call of solve.
        """
        super().__init__(graph, semantic)
        self.methods = ["dissatisfaction", "satisfaction", "utility"]
        self.aggs = ["sum", "min", "leximin"]

    def __str__(self):
        super().__str__()

    def voter_dissatisfaction(self, voter: Vote, extension: Extension):
        """Counts disagreements (opposite view points) between the voter and the extension"""
        e = ext_to_vec(extension, self.obaf.af.arguments)
        cpt = 0
        for name in self.obaf.af.arguments.keys():
            if voter[name] == -1 * e[name]:
                cpt = cpt + 1
        return cpt * -1

    def voter_satisfaction(self, voter: Vote, extension: Extension):
        """Counts agreements between the voter and the extension"""
        e = ext_to_vec(extension, self.obaf.af.arguments)
        cpt = 0
        for name in self.obaf.af.arguments.keys():
            if voter[name] == e[name]:
                cpt = cpt + 1
        return cpt

    def voter_satisfaction_abs(self, voter: Vote, extension: Extension):
        """Counts agreements, treating abstentions (0) as agreements"""
        e = ext_to_vec(extension, self.obaf.af.arguments)
        cpt = 0
        for name in self.obaf.af.arguments.keys():
            if (voter[name] == e[name]) or (voter[name] == 0):
                cpt = cpt + 1
        return cpt

    def voter_utility(self, voter: Vote, extension: Extension):
        """Net utility (satisfaction minus dissatisfaction)"""
        return self.voter_satisfaction(voter, extension) + self.voter_dissatisfaction(voter, extension)

    def sum_distance(self, extension, fun):
        """Aggregates scores using sum (utilitarian)"""
        method = getattr(self, fun, None)
        if method is not None and callable(method):
            tmp = []
            for v in self.obaf.votes:
                tmp.append(method(v, extension))
            # print(tmp)
            return sum(tmp)
        else:
            print("Invalid method name or method is not callable")

    def min_distance(self, extension, fun):
        """Aggregates scores using min (egalitarian)"""
        method = getattr(self, fun, None)
        if method is not None and callable(method):
            tmp = []
            for v in self.obaf.votes:
                tmp.append(method(v, extension))
            return min(tmp)
        else:
            print("Invalid method name or method is not callable")

    def leximin_distance(self, extension, fun: str):
        """Aggregates scores using leximin (returns sorted vector of scores)"""
        method = getattr(self, fun, None)
        if method is not None and callable(method):
            tmp = []
            for v in self.obaf.votes:
                tmp.append(method(v, extension))
            return sorted(tmp)
        else:
            print("Invalid method name or method is not callable")

    def argmax_distances(self, extensions, agg: str, meth: str):
        """Finds extensions that maximize the given aggregation method"""
        aggreg = getattr(self, agg, None)
        if aggreg is not None and callable(aggreg):
            res = {}
            for e in extensions:
                res[e] = aggreg(e, meth)
            if "leximin" not in agg:
                max_score = max(res.values())
                return [k for k,v in res.items() if v == max_score]
            else:
                for i in range(len(res[extensions[0]])):
                    leximax = -1100000
                    for extension, distance in res.items():
                        if distance[i] > leximax:
                            leximax = distance[i]
                    keys_to_remove = [key for key, distances in res.items() if distances[i] < leximax]
                    for key in keys_to_remove:
                        del res[key]
                    if len(res) == 1:
                        break
                return list(res.keys())
        else:
            print("Invalid method name or method is not callable")

    def add_votes_from_exts(self, votes: List[Extension]):
        for v in votes:
            self.obaf.add_vote_from_ext(v)

    def solve(self):
        """
        Computes best extensions for all combinations of aggregation (sum, min, etc.)
        and metric (satisfaction, utility, etc.) given in the init method
        """
        self.resulting_extensions = dict()
        extensions = self.obaf.af.get_extensions(self.semantic)
        for agg in self.aggs:
            self.resulting_extensions[agg] = dict()
            for method in self.methods:
                exts = self.argmax_distances(extensions, agg + "_distance",
                                             "voter_" + method)
                self.resulting_extensions[agg][method] = [Extension(x) for x in exts]

    def get_matrix(self, fun):
        """Returns a raw matrix of scores for all extensions vs aggregation methods, useful for printing"""
        res = list()
        for e in self.obaf.af.get_extensions(self.semantic):
            tmp = list()
            for agg in self.aggs:
                aggreg = getattr(self, agg+"_distance", None)
                tmp.append(aggreg(e, fun))
            res.append(tmp)
        return res

    def get_results_ids(self, extensions, method):
        """Helper to identify indices of optimal extensions for printing highlights"""
        if self.resulting_extensions is None:
            self.solve()

        res = []
        aggs = self.aggs
        for i in range(len(extensions)):
            for j in range(len(aggs)):
                for ext in self.resulting_extensions[aggs[j]][method]:
                    if extensions[i] == ext and len(ext) == len(extensions[i]):
                        res.append((i, j))
                        break
        return res

    def print_everything(self):
        """
        Prints a formatted comparison table of extensions and their scores across methods,
        highlighting the winners
        """
        if not self.obaf.votes:
            print("no votes")
            return

        methods = self.methods
        matricies = list()
        for m in methods:
            matricies.append(self.get_matrix("voter_"+m))

        ending = '\t\t\t'
        if len(matricies[0][0]) < 3:
            ending = '\t\t\t\t'

        for m in matricies:
            for l in m:
                l[2] = "".join([str(x) for x in l[2]])

        indecies = []
        for m in methods:
            indecies.append(self.get_results_ids(self.obaf.af.get_extensions(self.semantic), m))

        threshold = 4
        max = 24

        print(end='\t\t')
        for m in methods:
            print(m, end='\t\t'*3)
        print()

        print(end='\t\t')
        for m in ["sum|max|leximax", "sum|min|leximin", "sum|min|leximin"]:
            print(m, end='\t\t'*3)
        print()

        for d, s, u, p, i in zip(matricies[0], matricies[1], matricies[2], self.obaf.af.get_extensions(self.semantic),
                                     range(len(self.obaf.af.get_extensions(self.semantic)))):
            print(p, end='\t')

            for j in range(len(d)):
                if (i, j) in indecies[0]:
                    print('\033[43m ' + str(d[j]), end=", ")
                else:
                    print('\033[49m ' + str(d[j]), end=", ")
            print('\033[49m', end="")
            print(end='\t'*int((max-len(d[2]))/threshold))

            for j in range(len(s)):
                if (i, j) in indecies[1]:
                    print('\033[43m ' + str(s[j]), end=", ")
                else:
                    print('\033[49m ' + str(s[j]), end=", ")
            print('\033[49m', end="")
            print(end='\t'*int((max-len(s[2]))/threshold))

            for j in range(len(u)):
                if (i, j) in indecies[2]:
                    s = '\033[43m '
                else:
                    s = '\033[49m '
                print(s + str(u[j]), end=", ")
            print('\033[49m', end="")
            print()