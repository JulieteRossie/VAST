# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from math import floor
from typing import List
import numpy as np
from random import choice, shuffle
import itertools

from structure.vote import Vote
from structure.vote_generation_strategy import VoteGenerationStrategy
from structure.extension import Extension


class BasicStrategy(VoteGenerationStrategy):
    """Strategies to generate synthetic random votes based on normal, mean, or uniform distributions."""
    num_voters, semantic = None, None
    permutations = None

    def __init__(self, num_voters, semantic):
        self.num_voters = num_voters
        self.semantic = semantic
        self.reset_permutations()

    def generate_normal_votes(self, obaf, mean_perct, std_perct, gt: Extension, no_abs=True, is_consistent=False) -> Extension:
        """
        Generates votes using a normal distribution around a mean percentage of reliability

        :param obaf: the OBAF (AF + votes)
        :param mean_perct: mean voter reliability
        :param std_perct: STD voter reliability
        :param gt: the ground truth extension
        :param no_abs: the generated votes contain abstaintions or not
        :param is_consistent: the generated votes are consistent or not
        :return: returns the ground truth
        """
        obaf.remove_votes()
        if gt is None:
            gt = choice(obaf.get_extensions(self.semantic))
        self.get_permutations(obaf)
        mean_score = self.score_from_percentage(mean_perct, obaf, no_abs)
        for i in range(self.num_voters):
            if not is_consistent:
                score = int(np.random.normal(mean_score, std_perct))
                score = max(min(score, len(obaf.af.arguments)), -len(obaf.af.arguments))
                if score not in self.permutations.keys():
                    search_tab = self.permutations[score - 1] + self.permutations[score + 1]
                else:
                    search_tab = self.permutations[score]
                permutation = choice(search_tab)
                obaf.add_vote(self.vote_from_permutation(obaf, permutation, gt))
            else:
                raise NotImplementedError('Consistent normal votes not implemented.')
        return gt

    def generate_mean_votes(self, obaf, reliability, gt: Extension, no_abs=True, is_consistent=False) -> Extension:
        """
        Generates votes with varying levels of voter reliability dispersed around the mean

        :param obaf: the OBAF (AF + votes)
        :param reliability: mean reliability
        :param gt: the ground truth extension
        :param no_abs: the generated votes contain abstaintions or not
        :param is_consistent: the generated votes are consistent or not
        :return: returns the ground truth
        """
        if self.num_voters > 50:
            all_num_votes = [int(self.num_voters/20), int(self.num_voters/10), int(self.num_voters/5),
                             int(self.num_voters/2), int(self.num_voters)]
            first = all_num_votes[0]
            all_num_votes = [all_num_votes[i] - all_num_votes[i - 1] for i in range(1, len(all_num_votes))]
            all_num_votes.insert(0, first)
        else:
            all_num_votes = [int(self.num_voters)]
        obaf.remove_votes()
        if gt is None:
            gt = choice(obaf.get_extensions(self.semantic))
        self.get_permutations(obaf, no_abs)
        for num_votes in all_num_votes:
            percentages = self.get_mean_percentages(reliability, num_votes)
            for perc in percentages:
                score = self.score_from_percentage(perc, obaf, no_abs)
                if score not in self.permutations.keys():
                    search_tab = self.permutations[score-1] + self.permutations[score+1]
                else:
                    search_tab = self.permutations[score]
                vote = None
                permutation = choice(search_tab)
                values = list(permutation.values())
                i = 100
                while vote is None and i > 0:
                    shuffle(values)
                    shuffled_permutation = dict(zip(permutation.keys(), values))
                    if is_consistent:
                        vote = self.vote_from_permutation_consistent(obaf, shuffled_permutation, gt)
                    else:
                        vote = self.vote_from_permutation(obaf, shuffled_permutation, gt)
                    i -= 1
                if vote is not None:
                    obaf.add_vote(vote)
        return gt

    def generate_uniform_votes(self, obaf, reliability, gt: Extension, no_abs=True, is_consistent=False) -> Extension:
        """
        Generates votes uniformly distributed around a specific reliability score

        :param obaf: the OBAF (AF + votes)
        :param reliability: mean reliability
        :param gt: the ground truth extension
        :param no_abs: the generated votes contain abstaintions or not
        :param is_consistent: the generated votes are consistent or not
        :return: returns the ground truth
        """
        obaf.remove_votes()
        if gt is None:
            gt = choice(obaf.get_extensions(self.semantic))
        self.get_permutations(obaf, no_abs)
        score = self.score_from_percentage(reliability, obaf, no_abs)
        if score not in self.permutations.keys():
            search_tab = self.permutations[score-1] + self.permutations[score+1]
        else:
            search_tab = self.permutations[score]
        for i in range(self.num_voters):
            permutation = choice(search_tab)
            values = list(permutation.values())
            vote = None
            while vote is None:
                shuffle(values)
                suffled_permutation = dict(zip(permutation.keys(), values))
                if is_consistent:
                    vote = self.vote_from_permutation_consistent(obaf, suffled_permutation, gt)
                else:
                    vote = self.vote_from_permutation(obaf, suffled_permutation, gt)
            obaf.add_vote(vote)
        return gt

    def try_all(self, obaf, reliabilities, gt, compute_scores=True, generation_type="uniform") -> List:
        """Tests generation strategies across a list of reliability values"""
        obafs = []
        if generation_type == "uniform":
            function = self.generate_uniform_votes
        elif generation_type == "mean":
            function = self.generate_mean_votes
        elif generation_type == "normal":
            function = self.generate_normal_votes
        for dispersion in reliabilities:
            ground_truth = function(obaf, dispersion, gt, None)
            print("Ground truth = ", ground_truth)
            obaf.draw()
            if compute_scores:
                obafs.append(deepcopy(obaf))
        return obafs


    def reset_permutations(self):
        """Clears the cached vote permutations"""
        self.permutations = None

    def get_permutations(self, obaf, no_abs=True) -> None:
        """Computes or retrieves cached vote permutations, optionally filtering abstentions"""
        if self.permutations is None:
            self.compute_permutations(obaf)

            if no_abs:
                filtered_dict = {}

                for key, dict_list in self.permutations.items():
                    non_zero_dicts = [
                        d for d in dict_list
                        if not any(value == 0 for value in d.values())
                    ]

                    if non_zero_dicts:
                        filtered_dict[key] = non_zero_dicts

                self.permutations = filtered_dict

    @staticmethod
    def vote_from_permutation(obaf, permutation, gt):
        """Converts a permutation dictionary into a vote object relative to a ground truth"""
        arguments = obaf.af.arguments.keys()
        result = {
            key: (1 if key in gt.arguments else -1) * permutation[key]
            for key in arguments
        }
        return Vote('v' + str(obaf.vote_counter), dico=result)

    def vote_from_permutation_consistent(self, obaf, permutation, gt):
        """Converts a permutation to a vote, ensuring consistency with AF attacks"""
        vote = self.vote_from_permutation(obaf, permutation, gt)
        if not vote.is_consistent(obaf.af.attacks):
            vote = vote.get_consistent(obaf.af.attacks)
        return vote


    @staticmethod
    def score_from_percentage(dispersion, obaf, no_abs):
        """Computes the raw score corresponding to a percentage of agreement"""
        score = int((dispersion * len(obaf.af.arguments) * 2 / 100) - len(obaf.af.arguments))
        return score

    def compute_permutations(self, obaf):
        """Generates all possible score permutations based on vector composition"""
        self.permutations = dict()
        self.permutations[0] = list()
        tabs = self.create_vectors(0, obaf)
        for tab in tabs:
            self.permutations[0].append(dict(zip(obaf.af.arguments.keys(), tab)))

        for i in range(len(obaf.af.arguments) + 1)[1:]:
            self.permutations[i] = list()
            tabs = self.create_vectors(i, obaf)
            utabs = []
            for tab in tabs:
                self.permutations[i].append(dict(zip(obaf.af.arguments.keys(), tab)))

            self.permutations[-i] = list()
            tabs = self.create_vectors(-i, obaf)
            utabs = []
            for tab in tabs:
                self.permutations[-i].append(dict(zip(obaf.af.arguments.keys(), tab)))

    def create_vectors(self, score, obaf):
        """Helper to create base vectors for a given score"""
        res = []
        if score > 0:
            for i in range(score):
                res.append(1)
        elif score < 0:
            for i in range(abs(score)):
                res.append(-1)
        return self.complete_vectors(res, obaf)

    def complete_vectors(self, vector, obaf):
        """Completes vectors with cancelling pairs (1, -1) to maintain score"""
        pairs_left = floor((len(obaf.af.arguments) - len(vector))/2)
        vectors = [self.complete_zeros(vector, obaf)]
        for pair in range(pairs_left):
            vector.append(1)
            vector.append(-1)
            vectors.append(self.complete_zeros(vector, obaf))
        return vectors

    def complete_zeros(self, vector, obaf):
        """Pads the vector with zeros (abstentions) to match the number of arguments"""
        res = deepcopy(vector)
        while len(res) < len(obaf.af.arguments):
            res.append(0)
        return res

    def unique_permutations(self, s):
        """Generates unique string permutations via backtracking"""
        chars = sorted(s)
        result = []
        used = [False] * len(chars)

        def backtrack(path):
            if len(path) == len(chars):
                result.append(''.join(path))
                return

            for i in range(len(chars)):
                if used[i]:
                    continue
                if i > 0 and chars[i] == chars[i - 1] and not used[i - 1]:
                    continue

                used[i] = True
                path.append(chars[i])

                backtrack(path)

                used[i] = False
                path.pop()

        backtrack([])
        return result

    def compute_permutations2(self, obaf):
        """Alternative exhaustive permutation generation using itertools"""
        all_vectors = list(itertools.product([-1, 0, 1], repeat=len(obaf.af.arguments)))
        if (0,)*len(obaf.af.arguments) in all_vectors:
            all_vectors.remove((0,) * len(obaf.af.arguments))

        self.permutations = dict()
        self.permutations[0] = list()
        tabs = [vec for vec in all_vectors if sum(vec) == 0]
        for tab in tabs:
            self.permutations[0].append(dict(zip(obaf.af.arguments.keys(), tab)))
        for i in range(len(obaf.af.arguments) + 1)[1:]:
            self.permutations[i] = list()
            tabs = [vec for vec in all_vectors if sum(vec) == i]
            for tab in tabs:
                self.permutations[i].append(dict(zip(obaf.af.arguments.keys(), tab)))
            self.permutations[-i] = list()
            tabs = [vec for vec in all_vectors if sum(vec) == -i]
            for tab in tabs:
                self.permutations[-i].append(dict(zip(obaf.af.arguments.keys(), tab)))
