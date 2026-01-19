from copy import deepcopy
from itertools import combinations, chain
from cos.cos import COS
from typing import Union, List, Dict
from structure.opinion_based_af import OBAF
from structure.vote import Vote
from structure.extension import Extension


class ABSAF(COS):
    """
    Approval Based Social Argumentation Frameworks COS. This code is based on the code and paper in 10.5555/3635637.3662864
    """
    def __init__(self, graph: Union[str, OBAF], semantic, k: int = 1, representation_type="all", owa_param=0,
                 greedy=False):
        """
        Args:
            k: The size of the target set (number of extensions to select).
            representation_type: 'all' for the rep operator or 'defCore' for the core rep operator.
            owa_param: Parameter for the OWA operator (0=Utilitarian, 1=Harmonic, infty=Egalitarian, and None=MaxCov).
            greedy: If True, uses greedy approximation instead of exact search.
        """
        super().__init__(graph, semantic)
        self.k: int = k
        self._representation_type: str = representation_type
        self.owa_weight: float = owa_param
        self._greedy: bool = greedy

        # This will map each vote to the denominator needed to compute the defendable core,
        # so to avoid recomputing it.
        self._def_core_denominator = {}
        self.representation_options: List[str] = ["all", "defCore"]
        self.owa_weights: Dict[str, float] = {"utilitarian": 0, "harmonic": 1, "egalitarian": float("inf"), "MaxCov": None}

    def __str__(self):
        super().__str__()

    def solve(self):
        """Computes the best extensions for all configured OWA weights and types."""
        self.resulting_extensions = dict()

        for option in self.representation_options:
            self.set_representation_type(option)
            self.resulting_extensions[option] = dict()
            for m, weight in self.owa_weights.items():
                if weight is not None:
                    self.set_owa_value(weight)
                    self.resulting_extensions[option][m] = self.find_representation_owa(self.k)
                else:
                    self.resulting_extensions[option][m] = self.find_representation_max_cover(self.k)

    def print_everything(self):
        """Prints a comparison table of results for different OWA operators."""
        rounding = 2
        printable = ""
        all_prints = {}

        for option in self.representation_options:
            prints = {}
            self.set_representation_type(option)
            for S in combinations(self.obaf.af.get_extensions(self.semantic), int(self.k)):
                prints[str(S)] = []
                for m, weight in self.owa_weights.items():
                    self.set_owa_value(weight)
                    if weight is None:
                        score = self.representation_set_max_cover(S, 1)
                    else:
                        score = self.representation_set_owa(S)
                    prints[str(S)].append(score)
            all_prints[option] = deepcopy(prints)

        max_scores = {m: -float("inf") for m in self.owa_weights.keys()}

        for option, all_scores in all_prints.items():
            for extension, scores in all_scores.items():
                for idx, score in enumerate(scores):
                    m = list(self.owa_weights.keys())[idx]
                    if score > max_scores[m]:
                        max_scores[m] = score

        for option, all_scores in all_prints.items():
            printable += '\033[49m ' + "Using " + option + " representation:\n"
            printable += "\t" * 2 + "  ".join([m for m in self.owa_weights.keys()]) + "\n"
            for extension, scores in all_scores.items():
                spaces = 6
                printable += str(extension) + "\t"
                for idx, score in enumerate(scores):
                    m = list(self.owa_weights.keys())[idx]
                    if score == max_scores[m]:
                        printable += '\033[43m ' + str(round(score, rounding)) + ' \033[49m' + " " * (
                                    spaces + 4 - len(str(round(score, rounding))))
                    else:
                        printable += '\033[49m ' + str(round(score, rounding)) + ' \033[49m' + " " * (
                                    spaces + 4 - len(str(round(score, rounding))))
                printable += "\n"
        print(printable)

    def set_owa_value(self, owa_param):
        self.owa_weight = owa_param

    def set_representation_type(self, representation_type):
        assert representation_type in ("all", "defCore")
        self._representation_type = representation_type

    def set_greedy(self, greedy):
        self._greedy = greedy

    @staticmethod
    def _metricDistance(len_intersection, len_vote):
        """ Metric distance from a groundTruth and a Vote, used when generating votes with Mallows.
            the lenIntersection is the length of the intersection between groundTruth and the vote,
            and lenVote the length of the vote. """

        return len_vote - len_intersection

    def compute_def_core_data(self):
        """ When called, this function computes the data
            necessary to compute the def-core-representation of every voter and stores it for later. """
        for vote in self.obaf.votes:
            self.def_core_denominator(vote)

    def def_core_denominator(self, vote: Vote):
        if vote not in self._def_core_denominator:
            self._def_core_denominator[vote] = max(len(e.intersection(Extension(vote.yes()))) for e in
                                                   self.obaf.af.get_extensions(self.semantic))

        return self._def_core_denominator[vote]

    def representation_vote_ext(self, vote: Vote, extension: Extension):
        """ Given an approval ballot and a set of arguments, how much does this extension represent the ballot? """

        if self._representation_type == "all":
            if len(vote.yes()) == 0:
                return 0
            return len(vote.intersection(extension)) * 1.0 / len(vote.yes())
        elif self._representation_type == "defCore":
            if len(vote.yes()) == 0:
                return 0
            mu = self.def_core_denominator(vote)
            return len(set(vote.yes()).intersection(extension)) * 1.0 / mu if mu > 0 else 1
        else:
            raise Exception("The possible representation types are `all` and `defCore`.")

    def representation_vote_set(self, vote: Vote, s: List[Extension]):
        """ Given an approval ballot and a set of extensions (sets of arguments), how much does this set represent
        the ballot?"""
        return max(x for x in (self.representation_vote_ext(vote, extension) for extension in s) if x is not None)

    def representation_set_owa(self, s: List[Extension]):
        """ Given a set of approval ballots and a set of extensions (sets of arguments), how much does this set
        represent the ballots? In this method, we use the OWA score to assess it. """
        scores = sorted(self.representation_vote_set(vote, s) for vote in self.obaf.votes)
        weights = [1 / (i ** self.owa_weight) for i in range(1, len(self.obaf.votes) + 1)]
        return sum(a * i for a, i in zip(weights, scores))

    def find_representation_owa(self, n_extensions: int):
        """ Find a set of (nExtensions)-many preferred extensions that best represents the voters, using the OWA
        function"""
        if n_extensions >= len(self.obaf.af.get_extensions(self.semantic)):
            return list(self.obaf.af.get_extensions(self.semantic))
        else:
            best, best_score = None, -float("inf")
            if self._greedy:
                s = []
                while len(s) < n_extensions:
                    for e in self.obaf.af.get_extensions(self.semantic):
                        if e not in s:
                            score = self.representation_set_owa(s + [e])
                            if score > best_score:
                                best_score, best = score, e
                    s.append(best)

                return s
            else:
                best = []
                for s in combinations(self.obaf.af.get_extensions(self.semantic), n_extensions):
                    score = self.representation_set_owa(s)
                    if score > best_score:
                        best, best_score = [s[0]], score
                    elif score == best_score:
                        best.append(s[0])
                return best

    def representation_set_max_cover(self, s: List[Extension], alpha=1):
        """ Given a set of approval ballots and a set of extensions (sets of arguments), how much does this set represent the ballots?
            In this method, we use the (number of alpha-represented voters) score to asses it.

            By default, alpha=1."""
        return sum(1 for vote in self.obaf.votes if self.representation_vote_set(vote, s) >= alpha)

    def find_representation_max_cover(self, n_extensions: int, alpha=1):
        """ Find a set of nExtensions extensions that maximizes the alpha-represented voters.
        Note: alpha is 1 by default. """
        if n_extensions >= len(self.obaf.af.get_extensions(self.semantic)):
            return list(self.obaf.af.get_extensions(self.semantic))
        else:
            best, best_score = None, -float("inf")
            if self._greedy:
                S = []
                while len(S) < n_extensions:
                    for E in self.obaf.af.get_extensions(self.semantic):
                        if E not in S:
                            score = self.representation_set_max_cover(S + [E], alpha)
                            if score > best_score:
                                best_score, best = score, E
                    S.append(best)
                return S
            else:
                best = []
                for S in combinations(self.obaf.af.get_extensions(self.semantic), n_extensions):
                    score = self.representation_set_max_cover(S, alpha)
                    if score > best_score:
                        best, best_score = [S[0]], score
                    elif score == best_score:
                        best.append(S[0])
                return best

    def average_representation(self, s: List[Extension]):
        """ Given a set of extensions S, what is the average (across all voters) representation it gives? """
        return sum(self.representation_vote_set(vote, s) for vote in self.obaf.votes) / len(self.obaf.votes)

    def least_representation(self, s: List[Extension]):
        """ Given a set of extensions S, what is the min (across all voters) representation it gives? """
        return min(self.representation_vote_set(vote, s) for vote in self.obaf.votes)

    def perc_alpha_represented_voters(self, s: List[Extension], alpha):
        """ Given a set of extensions S, what is fraction of voters it alpha-represents? """
        return self.representation_set_max_cover(s, alpha) / len(self.obaf.votes)
