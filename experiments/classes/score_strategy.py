# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Union
import numpy as np

from structure.extension import Extension
from structure.arguments import Args
from structure.vote import Vote


def get_scoring_strategy(strategy_name):
    """Factory returning the specific scoring strategy instance by name"""
    if strategy_name == 'hamming':
        return HammingDistanceScore()
    elif strategy_name == 'jaccard':
        return JaccardScore()
    elif strategy_name == 'precision':
        return PrecisionScore()
    elif strategy_name == 'recall':
        return RecallScore()
    else:
        raise ValueError(f"Unknown scoring strategy: {strategy_name}")


class ScoringStrategy(ABC):
    """Abstract base class for all scoring metrics"""

    @abstractmethod
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args: Optional[Args] = None) -> float:
        """Computes the score of an extension (or vote) against a ground truth"""
        pass

    def compute_scores(self, extensions: List[Extension], ground_truth: Extension, method: str = "skeptical",
                       args: Optional[Args] = None) -> float:
        """Aggregates scores for a set of extensions using the specified method (sum, skeptical, etc.)"""
        assert type(ground_truth) is Extension
        if len(extensions) == 1 and method != "none":
            return self.compute_score(extensions[0], ground_truth, args)
        else:
            if method == "sum":
                scores = []
                for extension in extensions:
                    scores.append(self.compute_score(extension, ground_truth, args))
                return np.mean(scores)
            elif method == "skeptical":
                extension = self.get_skeptical(extensions)
                return self.compute_score(extension, ground_truth, args)
            elif method == "doubleskeptical":
                extension = self.get_doubleskeptical(extensions, args)
                return self.compute_score(extension, ground_truth, args)
            elif method == "none":
                return self.compute_score(extensions, ground_truth, args)
            else:
                raise ValueError(f"compute_scores: Unknown method: {method}")

    @staticmethod
    def get_skeptical(extensions: List[Extension]) -> Extension:
        """Returns the intersection of all provided extensions"""
        res = deepcopy(extensions[0])
        for extension in extensions:
            res = res.intersection(extension)
        return res

    @staticmethod
    def get_doubleskeptical(extensions: List[Extension], args: Args) -> Vote:
        """Returns a Vote reflecting unanimous agreement (yes), rejection (no), the other arguments will be zero"""
        yesses = []
        noes = []
        zeros = []
        for arg in args.keys():
            tmp = sum([1 for ext in extensions if arg in ext])
            if tmp == len(extensions):
                yesses.append(arg)
            elif tmp == 0:
                noes.append(arg)
            else:
                zeros.append(arg)
        return Vote("v", frozenset(yesses), frozenset(noes), frozenset(zeros))


class SimpleScore(ScoringStrategy):
    """Scoring based on inclusion of ground truth scaled by extension size"""
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args: Optional[Args] = None) -> float:
        if ground_truth in extension:
            return 1 / len(extension) * 100
        else:
            return 0


class SimilarityScore(ScoringStrategy):
    """Scoring based on L1 vector similarity"""
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args=None) -> float:
        assert (args is not None)
        vec_ground_truth = np.array(ground_truth.get_vector(args.keys()))
        vec_extension = np.array(extension.get_vector(args.keys()))
        score = len(args.keys()) - np.linalg.norm((vec_extension - vec_ground_truth), ord=1)
        return (score + len(args)) * 100 / (2 * len(args))


class HammingDistanceScore(ScoringStrategy):
    """Scoring based on Hamming distance"""
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args=None) -> float:
        return extension.hamming_distance(ground_truth)


class JaccardScore(ScoringStrategy):
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args=None) -> float:
        intersection = ground_truth.intersection(extension)
        if type(extension) != Extension:
            extension = Extension(extension.yes())
        union = ground_truth.union(extension)
        return len(intersection) / len(union) * 100


class PrecisionScore(ScoringStrategy):
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args=None) -> float:
        intersection = set(ground_truth).intersection(set(extension))
        if len(intersection) == 0:
            return 0
        return len(intersection) / len(extension)


class RecallScore(ScoringStrategy):
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args=None) -> float:
        intersection = set(ground_truth).intersection(set(extension))
        return len(intersection) / len(ground_truth)

class AbsafScore(ScoringStrategy):
    def compute_score(self, extension: Union[Extension, Vote, List[Extension]], ground_truth: Extension, args= None) -> float:
        if len(extension.intersection(ground_truth)) == 0:
            return 0
        return len(extension.intersection(ground_truth)) * 1.0 / len(extension)