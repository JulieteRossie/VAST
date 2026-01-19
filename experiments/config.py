# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math


distribution_params = {
    "uniform": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "normal": {
        "mean": [0.0, 0.2, 0.4, 0.6, 0.8, 1],
        "std": [0.2, 0.4, 0.6, 0.8]
    }
}

vote_params = {
    "semantics": ["pref"],
    "vote_types": {
        "ABSAF_vote": {
            "metric": "dispersion",
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        "Our_vote": {
            "metric": "reliability",
            "values": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
    }
}

def build_watts_strogatz_config():
    num_args = [5, 10, 15, 20, 30]
    prob_cycles = [0.2, 0.5, 0.8]
    prob_rewiring = [0.2, 0.5, 0.8]
    even_k = lambda n: [k for k in range(2, n) if k % 2 == 0]
    max_m = {
        "5": [k for k in range(2, 5) if k % 2 == 0],
        "10": [k for k in range(2, 10) if k % 2 == 0],
        "15": [k for k in range(2, 15) if k % 2 == 0],
        "20": [k for k in range(2, 20) if k % 2 == 0],
        "30": [k for k in range(2, 30) if k % 2 == 0],
    }

    return {
        "num_args": num_args,
        "graph_generation_metrics": {
            "prob_cycles": prob_cycles,
            "prob_rewiring": prob_rewiring,
            # "k_nearest_neighbor": lambda n: even_k(n)[len(even_k(n))//2:],
            "k_nearest_neighbor": lambda n: even_k(n),
            "number_of_afs": lambda n: math.ceil((600)/(len(prob_cycles)*
                                                        len(prob_rewiring)*len(max_m[str(n)]))),
        },
        "graph_code": [f"AF_{i}" for i in range(1, 101)],
    }

data_structure = {
    "BarabasiAlbert": {
        "num_args": [5, 10, 15, 20, 30, 40],
        "graph_generation_metrics": {
            "prob_cycles": [0.1, 0.25, 0.5, 0.75, 0.9],
        },
        "graph_code": [f"AF_{i}" for i in range(1, 101)]
    },
    "Gilbert": {
        "num_args": [5, 10, 15, 20, 30, 40],
        "graph_generation_metrics": {
            "prob_attacks": [0.1, 0.25, 0.5, 0.75, 0.9],
        },
        "graph_code": [f"AF_{i}" for i in range(1, 101)]
    },
    "WattsStrogatz": build_watts_strogatz_config()
}