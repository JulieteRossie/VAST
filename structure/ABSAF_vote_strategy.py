# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.special import comb
from collections import defaultdict
from random import choice, sample

from structure.vote_generation_strategy import VoteGenerationStrategy
from structure.extension import Extension


def _metric_distance(len_intersection, len_vote):
    """Metric distance for Mallows: absolute difference between vote length and intersection length."""
    return abs(len_vote - len_intersection)


class ABSAFVoteStrategy(VoteGenerationStrategy):
    """
    Vote generation strategy based on Mallows model using a specific ground truth extension.
    Based on the approach in: https://arxiv.org/pdf/2207.01140.pdf
    """
    num_voters, semantic = None, None

    def __init__(self, num_voters, semantic):
        self.num_voters = num_voters
        self.semantic = semantic

    def try_all(self, obaf, dispersions, gt=None, generation_type="uniform"):
        if generation_type == "uniform":
            for dispersion in reliabilities:
                ground_truths, probabilities, values, externals = self.prep_vote_generation(obaf, dispersion, gt, self.semantic)
                groundTruths = self.standard_votes(obaf, self.num_voters, ground_truths, probabilities, values, externals)
                print(groundTruths)
                obaf.draw()
        else:
            raise NotImplementedError("not implemented")

    def generate_uniform_votes(self, obaf, dispersion, gt=None) -> Extension:
        ground_truths, probabilities, values, externals = self.prep_vote_generation(obaf, dispersion, gt, self.semantic)
        return self.standard_votes(obaf, self.num_voters, ground_truths, probabilities, values, externals)

    def generate_normal_votes(self, obaf, mean, std, gt=None) -> Extension:
        raise NotImplementedError("This method is not yet implemented.")

    def generate_mean_votes(self, obaf, mean, std, gt=None) -> Extension:
        raise NotImplementedError("This method is not yet implemented.")

    def standard_votes(self, obaf, voters_per_ground_truth, ground_truth, probabilities, values, externals):
        """Generates votes by sampling from the pre-calculated probability distribution."""
        obaf.remove_votes()
        for _ in range(voters_per_ground_truth):
            v = np.random.choice(values, p=probabilities)
            x, y = map(int, v.split("#"))
            inside = sample(ground_truth.get_args(), x)
            outside = sample(externals, y - x)
            vote = frozenset(inside + outside)
            obaf.add_vote_from_ext(Extension(vote))

        return ground_truth

    def prep_vote_generation(self, obaf, dispersion, gt, semantic):
        """
        Prepares probabilities for the Mallows model.
        Returns the ground truth used, probabilities list, value keys (intersection#size), and external args.
        """

        # following https://arxiv.org/pdf/2207.01140.pdf
        if gt is None:
            if len(obaf.get_extensions(semantic)) <= 1:
                ground_truth = obaf.get_extensions(semantic)

            else:
                ground_truth = choice(obaf.get_extensions(semantic))
        else:
            ground_truth = gt

        m = len(obaf.af.arguments)
        externals = [a for a in obaf.af.arguments if a not in ground_truth]

        f = defaultdict(float)
        z = len(ground_truth)
        Z = 0

        values = []

        for x in range(len(ground_truth) + 1):
            for y in range(x, m + 1):
                if f"{x}#{y}" in values:
                    return
                assert f"{x}#{y}" not in values
                values.append(f"{x}#{y}")

                f[(x, y)] = 0 if y == 0 else (
                        comb(z, x) * comb(m - z, y - x) * (dispersion ** _metric_distance(x, y)))
                Z += f[(x, y)]

        probabilities = []
        for v in values:
            x, y = map(int, v.split("#"))
            probabilities.append(f[(x, y)] / Z)

        return ground_truth, probabilities, values, externals
