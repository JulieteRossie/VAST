# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import random

from structure.extension import Extension


class VoteGenerationStrategy(ABC):
    """Abstract base class for vote generation strategies"""
    @abstractmethod
    def generate_normal_votes(self, obaf, mean_perct, std_perct, gt: Extension) -> Extension:
        """Abstract method for generating votes."""
        pass

    @abstractmethod
    def generate_uniform_votes(self, obaf, reliability, gt: Extension) -> Extension:
        """Abstract method for generating votes."""
        pass

    @abstractmethod
    def generate_mean_votes(self, obaf, reliability, gt: Extension) -> Extension:
        """Abstract method for generating votes."""
        pass

    @abstractmethod
    def try_all(self, obaf, reliabilities, gt, generation_type="uniform") -> None:
        """Tests the strategy across a range of reliability values"""
        pass

    @staticmethod
    def get_mean_percentages(target_mean, num_votes, interval=25, seed=None):
        """Generates a list of percentages that average exactly to the target mean"""
        if target_mean == 0:
            return [0] * num_votes
        if target_mean == 100:
            return [100] * num_votes
        # Ensure target_score doesn't exceed 0 and 100
        safe_interval = min(min(target_mean, interval), min(interval, 100 - target_mean))
        assert safe_interval >= 0

        distances = [random.randint(-safe_interval, safe_interval) for _ in range(num_votes)]
        distance_to_target = sum(distances)
        indecies = list(range(num_votes))

        while distance_to_target != 0:
            idx = random.choice(indecies)
            if distance_to_target > 0:
                interval_wiggle_room = safe_interval + distances[idx]
                if interval_wiggle_room == 0:
                    indecies.remove(idx)
                    continue
                alteration = random.randint(1, min(abs(distance_to_target), interval_wiggle_room))
                assert alteration >= 0
                distances[idx] -= alteration
                assert sum(distances) >= 0
            elif distance_to_target < 0:
                interval_wiggle_room = safe_interval - distances[idx]
                if interval_wiggle_room == 0:
                    indecies.remove(idx)
                    continue
                alteration = random.randint(1, min(abs(distance_to_target), interval_wiggle_room))
                assert alteration >= 0
                distances[idx] += alteration
                assert sum(distances) <= 0
            distance_to_target = sum(distances)

        scores = [target_mean + dist for dist in distances]
        assert all(score >= 0 and score <= 100 for score in scores)
        random.shuffle(scores)
        return scores
