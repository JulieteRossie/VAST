# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import project_config as config
from structure.opinion_based_af import OBAF
from cos.collective_satisfaction_semantics import CSS
from cos.attack_removal_semantics import ARS
from cos.judgement_aggregation import JA
from cos.approval_based_social_af import ABSAF
from cos.majority_rule import MR
from experiments.classes.generate_experiments_results import ExperimentRunner


# --- 1. DEMO FUNCTIONALITY (to try out the different cos functions on the examples) ---

def run_single_file_demo(filename, semantic="pref", visualize=False):
    """
    Loads an APX file and runs all COS methods on it.
    """
    path = config.FrameworksPath
    file_path = path / filename
    if not file_path.exists():
        print(f"Error: File {filename} not found.")
        return

    print(f"Loading: {filename}")
    g = OBAF()
    g.populate(filename=filename, path=path, read_votes=True, wrong_arg_name_format=True, isfile=True)

    if visualize:
        print("Displaying Graph...")
        g.draw()

    print(f"\n{'=' * 40}")
    print(f"Running Aggregations (Semantic: {semantic})")
    print(f"{'=' * 40}\n")

    # 1. CSS
    print("--- Collective Satisfaction Semantics (CSS) ---")
    agg = CSS(g, semantic)
    agg.solve()
    agg.print_everything()
    print()

    # 2. ARS
    print("--- Attack Removal Semantics (ARS) ---")
    soc = ARS(g, semantic)
    soc.solve()
    soc.print_everything()
    print()

    # 3. LA
    print("--- Labelling Aggregation (JA) ---")
    pigozzi = JA(g, semantic)
    pigozzi.solve()
    pigozzi.print_everything()
    print()

    # 4. ABSAF
    print("--- Approval Ballot Social AFs (ABSAF) ---")
    new = ABSAF(g, semantic)
    new.solve()
    new.print_everything()
    print()

    # 5. Majority rule
    print("--- Majority Rule (MR) ---")
    merg = MR(g, semantic)
    merg.solve()
    merg.print_everything()
    print()


# --- 2. EXPERIMENT REPLICATION (to replicate the results from the paper) ---

def run_experiment_suite(db_name, graph_type, semantic, num_args):
    """
    Replicates results from the paper
    """
    print(f"Starting Experiment Runner on DB: VAST_dataset")
    print(f"Results will be saved in DB: {db_name}")
    print(f"Parameters: Graph={graph_type}, Semantic={semantic}, Args={num_args}")

    experiment = ExperimentRunner(db_name)
    experiment.run_full_experiment(
        "VAST_dataset.db",
        semantic=semantic,
        graph_type=graph_type,
        votes_distribution="mean",
        num_arguments=num_args
    )
    print("Experiment completed.")


# --- 3. PLOTTING UTILITIES ---

def run_plotting_suite(db_path, graph_type, num_votes, vote_gen_type):
    """
    Generates the standard plots from the experiment database.
    """
    print(f"Generating plots from {db_path}...")

    excludes = ["w-abs", "dissatisfaction", "satisfaction", "TestingEmpty",
                "TestingAll", "TestingRandom", "maxcov", "min", "harmonic",
                "skeptical", "credulous", "defCore"]

    # This calls your existing logic for plotting
    experiment = ExperimentRunner(db_path)

    # Example plot configuration (adapted from your original code)
    for semantic in ["pref"]:
        score_strategy = "SimpleScore"
        score_metric = "none"

        print(f"Plotting: {semantic} / {score_strategy}")

        fixed_params = {
            "score_strategy": score_strategy,
            "score_metric": score_metric,
            "num_votes": num_votes,
            "semantic": "pref",
            "vote_type": "nozeros",
            "graph_type": graph_type
        }

        # General paper plot with pref semantics with accuracy metric
        experiment.plot_score_with_confidence(
            x_axis="metric_value",
            y_axis="score",
            exclude_keywords=excludes,
            num_afs=None,
            fixed_params=fixed_params,
            vote_distrib="mean",
            vote_value_cuttoff=40,
            title=f"Experiment Results: {graph_type}",
            vote_generator_type=vote_gen_type,
            invert_x=False
        )

        fixed_params["semantic"] = "comp"

        # General paper plot with comp semantics with accuracy metric
        experiment.plot_score_with_confidence(
            x_axis="metric_value",
            y_axis="score",
            exclude_keywords=excludes,
            num_afs=None,
            fixed_params=fixed_params,
            vote_distrib="mean",
            vote_value_cuttoff=40,
            title=f"Experiment Results: {graph_type}",
            vote_generator_type=vote_gen_type,
            invert_x=False
        )

        fixed_params["semantic"] = "pref"
        fixed_params["score_strategy"] = "SimilarityScore"
        fixed_params["score_metric"] = "doubleskeptical"

        # General paper plot with pref semantics with similarity metric
        experiment.plot_score_with_confidence(
            x_axis="metric_value",
            y_axis="score",
            exclude_keywords=excludes,
            num_afs=None,
            fixed_params=fixed_params,
            vote_distrib="mean",
            vote_value_cuttoff=40,
            title=f"Experiment Results: {graph_type}",
            vote_generator_type=vote_gen_type,
            invert_x=False
        )

        fixed_params["score_strategy"] = "SimpleScore"
        fixed_params["score_metric"] = "none"
        reliability = 60
        fixed_params.pop("num_votes")

        # number of votes plot compared to score with a fixed reliability
        title=f"Number of votes evolution for dipersion {reliability}"
        experiment.plot_score_by_parameter2(
            x_axis="num_votes", y_axis="score",
            exclude_keywords=excludes, score_strategy=fixed_params["score_strategy"],
            fixed_params=fixed_params, vote_value_cuttoff=40,
            vote_generator_type=vote_gen_type,
            title=title, metric_value=60, num_bins=9, bin_mode="equal_count")

        # number of extension plot compared to score with a fixed reliability
        title=f"Number of extensions evolution for dipersion {reliability}"
        experiment.plot_score_by_parameter2(
            x_axis="num_extensions", y_axis="score",
            exclude_keywords=excludes, score_strategy=fixed_params["score_strategy"],
            fixed_params=fixed_params, vote_value_cuttoff=40,
            vote_generator_type=vote_gen_type,
            title=title, metric_value=60, num_bins=9, bin_mode="equal_count")


# --- MAIN ENTRY POINT ---

def main():
    parser = argparse.ArgumentParser(description="Opinion-Based AF Experiment Framework")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo arguments
    parser_demo = subparsers.add_parser("demo", help="Run aggregations on a single file")
    parser_demo.add_argument("--file", type=str, help="Path to the .apx file")
    parser_demo.add_argument("--semantics", type=str, default="pref", help="Semantics to use (pref, comp, etc.)")
    parser_demo.add_argument("--viz", action="store_true", help="Visualize the graph")

    # Replicating experiment arguments
    parser_exp = subparsers.add_parser("replicate", help="Run the full experiment suite")
    parser_exp.add_argument("--db", type=str, default="results.db", help="Results Database Name")
    parser_exp.add_argument("--graph", type=str, default=None, help="Graph Type")
    parser_exp.add_argument("--semantics", type=str, default=None, help="Semantics")
    parser_exp.add_argument("--args", type=int, default=None, help="Number of Arguments")

    # Plotting arguments
    parser_plot = subparsers.add_parser("plot", help="Generate plots from results")
    parser_plot.add_argument("--db", type=str, help="Source Database Name")
    parser_plot.add_argument("--graph", type=str, default="BarabasiAlbert", help="Graph Type")
    parser_plot.add_argument("--votes", type=int, default=50, help="Number of votes")
    parser_plot.add_argument("--gen_type", type=str, default="Our_vote", help="Vote generator type")

    args = parser.parse_args()

    if args.command == "demo":
        run_single_file_demo(args.file, args.semantics, args.viz)

    elif args.command == "replicate":
        run_experiment_suite(args.db, args.graph, args.semantics, args.args)

    elif args.command == "plot":
        run_plotting_suite(args.db, args.graph, args.votes, args.gen_type)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()