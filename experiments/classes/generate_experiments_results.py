# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func, create_engine, Column, Integer, String, Float, text, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import random

from structure.opinion_based_af import OBAF
from experiments.classes.VASThandler import VASTHandler, GraphModel, VoteModel
from cos.majority_rule import MR
from cos.attack_removal_semantics import ARS
from cos.collective_satisfaction_semantics import CSS
from cos.judgement_aggregation import JA
from cos.approval_based_social_af import ABSAF
from cos.baselines import AllExtensionsBaseline
from experiments.classes.score_strategy import SimpleScore, SimilarityScore, AbsafScore

dark_color = "#1A1AFF"
medium_dark_color = "#95e000"
medium_color = "#FF1AB3"
medium_color2 = "#1AB2FF"
medium_light_color = "#FFB31A"
light_color = "#b6c2ba"

final_style_mapping = {
                "CSS: sum & utility": {"color": f"{dark_color}", "linestyle": "-", "marker": "o"},
                "CSS: min & utility": {"color": f"{dark_color}", "linestyle": ":", "marker": "s"},
                "CSS: leximin & utility": {"color": f"{dark_color}", "linestyle": "--", "marker": "^"},
                "Majority Rule": {"color": f"{medium_color2}", "linestyle": "-", "marker": ""},
                "ABSAF: all & utilitarian": {"color": f"{medium_light_color}", "linestyle": "-", "marker": "s"},
                "ABSAF: all & harmonic": {"color": f"{medium_light_color}", "linestyle": ":", "marker": "s"},
                "ABSAF: all & egalitarian": {"color": f"{medium_light_color}", "linestyle": "--", "marker": "v"},
                "ABSAF: all & MaxCov": {"color": f"{medium_light_color}", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "d"},
                "ABSAF: defCore & utilitarian": {"color": "#E377C2", "linestyle": ":", "marker": "s"},
                "ABSAF: defCore & harmonic": {"color": "#E377C2", "linestyle": "-", "marker": "o"},
                "ABSAF: defCore & egalitarian": {"color": "#E377C2", "linestyle": "--", "marker": "^"},
                "ABSAF: defCore & MaxCov": {"color": "#E377C2", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "d"},
                "ARS": {"color": f"{medium_color}", "linestyle": "-", "marker": "X"},
                "ARS: w-abs": {"color": "#FF7F0E", "linestyle": "--", "marker": "^"},
                "ARS: grounded": {"color": "#FF7F0E", "linestyle": ":", "marker": "s"},
                "LA: super-credulous": {"color": f"{medium_dark_color}", "linestyle": "-", "marker": "*"},
                "All extensions": {"color": f"{light_color}", "linestyle": ":", "marker": ""},
                "CSS: sum & satisfaction": {"color": "#2CA02C", "linestyle": ":", "marker": "s"},
                "CSS: min & satisfaction": {"color": "#2CA02C", "linestyle": "-", "marker": "o"},
                "CSS: leximin & satisfaction": {"color": "#2CA02C", "linestyle": "--", "marker": "^"},
                "CSS: sum & dissatisfaction": {"color": "#D02D2E", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "d"},
                "CSS: min & dissatisfaction": {"color": "#D02D2E", "linestyle": ":", "marker": "s"},
                "CSS: leximin & dissatisfaction": {"color": "#D02D2E", "linestyle": "-", "marker": "o"},
            }


_absaf_vote_type = "ABSAF_votes"
_our_vote_type = "Our_votes"

Base = declarative_base()

class ExperimentTableModel(Base):
    """SQLAlchemy model for storing experiment results"""
    __tablename__ = 'experiment_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_type = Column(String, nullable=False)
    num_arguments = Column(Integer, nullable=False)
    graph_generation_metrics = Column(JSON, nullable=False)
    graph_id = Column(Integer, nullable=False)
    semantic = Column(String, nullable=False)
    num_extensions = Column(Integer, nullable=False)
    ground_truth = Column(String, nullable=False)
    vote_generator_type = Column(String, nullable=False)
    vote_type = Column(String, nullable=False)
    vote_distribution = Column(String, nullable=False)
    vote_metric = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    vote_id = Column(Integer, nullable=False)
    aggregation_name = Column(String, nullable=False)
    agg_metric_1 = Column(String, nullable=False)
    agg_metric_2 = Column(String, nullable=False)
    num_votes = Column(Integer, nullable=False)
    score_strategy = Column(String, nullable=False)
    score_metric = Column(Integer, nullable=False)
    resulting_extension = Column(String, nullable=False)
    score = Column(Float, nullable=False)


class ExperimentRunner:
    def __init__(self, db_name, aggregations=None, score_strategies=None, score_metrics=None, num_votes=None):
        """Initialize the experiment runner"""
        self.db_path = db_name
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        if aggregations is None:
            aggregations = ["JA", "ARS", "CSS", "ABSAF", "MR", "AllExtensionsBaseline"]
        if score_strategies is None:
            score_strategies = ["SimpleScore", "SimilarityScore" , "AbsafScore"]
        if score_metrics is None:
            score_metrics = ["sum", "doubleskeptical", "none"]
        if num_votes is None:
            num_votes = [5, 10, 20, 50, 100]
        self.aggregations = aggregations
        self.score_strategies = score_strategies
        self.score_metrics = score_metrics
        self.num_votes = num_votes

    def process_single_vote(self, vote, other_db) -> List[dict]:
        score_batch = list()

        g = OBAF()
        g.populate(text=vote.graph.graph_apx, isfile=False, read_votes=False)
        g.af._solution = other_db.json_desseriasable_extensions_dict(vote.graph.extensions)

        for num_v in self.num_votes:
            g.repopulate_vote_files(text=vote.votes, num_votes=num_v)

            for agg_name in self.aggregations:
                agg_method_class = globals()[agg_name]
                agg = agg_method_class(g, vote.semantic)
                if vote.semantic == "pref":
                    num_ext = vote.graph.pref_num_extensions
                    gt = vote.graph.pref_ground_truth
                elif vote.semantic == "comp":
                    num_ext = vote.graph.comp_num_extensions
                    gt = vote.graph.comp_ground_truth
                agg.solve()

                for score_strategy_name in self.score_strategies:
                    score_strategy_class = globals()[score_strategy_name]
                    strategy = score_strategy_class()

                    for score_metric in self.score_metrics:
                        if score_strategy_name != "SimpleScore" and score_metric == "none":
                            continue
                        elif score_strategy_name == "SimpleScore" and score_metric != "none":
                            continue
                        elif score_strategy_name == "AbsafScore" and score_metric != "sum":
                            continue

                        gt_ext = other_db.json_desseriasanle_extension(gt)
                        for key, params in agg.resulting_extensions.items():
                            for subkey, extensions in params.items():
                                if len(extensions) == 0:
                                    score = 0
                                else:
                                    score = strategy.compute_scores(
                                        extensions=extensions,
                                        ground_truth=gt_ext,
                                        args=g.af.arguments,
                                        method=score_metric
                                    )
                                tmp = ','.join([str(x) for x in agg.resulting_extensions[key][subkey]])
                                score_batch.append({
                                    'graph_type': vote.graph.graph_type,
                                    'num_arguments': vote.graph.num_arguments,
                                    'graph_generation_metrics': vote.graph.graph_generation_metrics,
                                    'graph_id': vote.graph.id,
                                    'semantic': vote.semantic,
                                    'num_extensions': int(num_ext),
                                    'ground_truth': '-'.join(gt),
                                    'vote_generator_type': vote.vote_generator_type,
                                    'vote_type': vote.vote_type,
                                    'vote_distribution': vote.votes_distribution,
                                    'vote_metric': vote.vote_metric,
                                    'metric_value': vote.vote_metric_value,
                                    'vote_id': vote.id,
                                    'aggregation_name': agg_name,
                                    'agg_metric_1': key,
                                    'agg_metric_2': subkey,
                                    'num_votes': num_v,
                                    'score_strategy': score_strategy_name,
                                    'score_metric': score_metric,
                                    'resulting_extension': tmp,
                                    'score': score
                                })
        return score_batch

    def run_full_experiment(self, database_name: str, semantic: str, num_arguments: int,
                            graph_type: str, **query_filters):
        print("Running full experiment started")
        db_manager = VASTHandler(database_name)
        session = db_manager.Session()

        query = session.query(VoteModel).join(VoteModel.graph)

        query = query.filter(GraphModel.graph_type == graph_type)
        query = query.filter(GraphModel.num_arguments == num_arguments)
        query = query.filter(VoteModel.semantic == semantic)

        votes = (
            query.all()
        )

        session = self.Session()
        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for vote in votes:
                    future = executor.submit(self.process_single_vote, vote, db_manager)
                    futures.append(future)

                for future in tqdm(futures):
                    try:
                        score_batch = future.result()
                        if score_batch:
                            self.insert_scores_batch(score_batch)
                    except Exception as e:
                        print(f"run_full_experiment: Error processing vote: {e}")

        except SQLAlchemyError as e:
            print(f"run_full_experiment: Error while generating votes: {e}")
            session.rollback()
        finally:
            session.close()

    def insert_scores_batch(self, votes: list[dict]) -> None:
        """Insert multiple votes into the ExperimentTableModel table in batch"""
        session = self.Session()
        try:
            new_votes = [
                ExperimentTableModel(
                    graph_type=v['graph_type'],
                    num_arguments=v['num_arguments'],
                    graph_generation_metrics=v['graph_generation_metrics'],
                    graph_id=v['graph_id'],
                    semantic=v['semantic'],
                    ground_truth=v['ground_truth'],
                    num_extensions=v['num_extensions'],
                    vote_generator_type=v['vote_generator_type'],
                    vote_type=v['vote_type'],
                    vote_distribution=v['vote_distribution'],
                    vote_metric=v['vote_metric'],
                    metric_value=v['metric_value'],
                    vote_id=v['vote_id'],
                    aggregation_name=v['aggregation_name'],
                    agg_metric_1=v['agg_metric_1'],
                    agg_metric_2=v['agg_metric_2'],
                    num_votes=v['num_votes'],
                    score_strategy=v['score_strategy'],
                    score_metric=v['score_metric'],
                    resulting_extension=v['resulting_extension'],
                    score=v['score']
                )
                for v in votes
            ]

            session.bulk_save_objects(new_votes)
            session.commit()

        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def insert_score(self,
                     graph_type: str,
                     num_arguments: int,
                     graph_generation_metrics: dict,
                     graph_id: int,
                     semantic: str,
                     num_extensions: int,
                     ground_truth: str,
                     vote_generator_type: str,
                     vote_type: str,
                     vote_distribution: str,
                     vote_metric: str,
                     metric_value: float,
                     vote_id: int,
                     aggregation_name: str,
                     agg_metric_1: str,
                     agg_metric_2: str,
                     num_votes: int,
                     score_strategy: str,
                     score_metric: int,
                     resulting_extension: str,
                     score: float) -> int:

        session = self.Session()
        try:
            existing_score = session.query(ExperimentTableModel).filter_by(
                graph_type=graph_type,
                num_arguments=num_arguments,
                graph_generation_metrics=graph_generation_metrics,
                graph_id=graph_id,
                semantic=semantic,
                num_extensions=num_extensions,
                ground_truth=ground_truth,
                vote_generator_type=vote_generator_type,
                vote_type=vote_type,
                vote_distribution=vote_distribution,
                vote_metric=vote_metric,
                vote_id=vote_id,
                metric_value=metric_value,
                aggregation_name=aggregation_name,
                agg_metric_1=agg_metric_1,
                agg_metric_2=agg_metric_2,
                num_votes=num_votes,
                score_strategy=score_strategy,
                score_metric=score_metric,
                resulting_extension=resulting_extension,
                score=score
            ).first()

            if existing_score:
                return existing_score.id

            result = ExperimentTableModel(
                graph_type=graph_type,
                num_arguments=num_arguments,
                graph_generation_metrics=graph_generation_metrics,
                graph_id=graph_id,
                semantic=semantic,
                ground_truth=ground_truth,
                num_extensions=num_extensions,
                vote_generator_type=vote_generator_type,
                vote_type=vote_type,
                vote_distribution=vote_distribution,
                vote_metric=vote_metric,
                metric_value=metric_value,
                vote_id=vote_id,
                aggregation_name=aggregation_name,
                agg_metric_1=agg_metric_1,
                agg_metric_2=agg_metric_2,
                num_votes=num_votes,
                score_strategy=score_strategy,
                score_metric=score_metric,
                resulting_extension=resulting_extension,
                score=score
            )
            session.add(result)
            session.commit()
            result_id = result.id
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

        return result_id

    def query_votes_raw(self, **kwargs):
        """Query the votes table and return raw SQLAlchemy objects"""
        session = self.Session()
        try:
            query = session.query(ExperimentTableModel)
            for key, value in kwargs.items():
                query = query.filter(getattr(ExperimentTableModel, key) == value)
            return query.all()
        finally:
            session.close()

    def plot_score(self, x_axis, y_axis, vote_generator_type, vote_distrib, num_afs, exclude_keywords=None,
                   fixed_params=None, vote_value_cuttoff=0, title=None, invert_x=False):
        session = self.Session()

        try:
            if exclude_keywords is None:
                exclude_keywords = []
            if fixed_params is None:
                fixed_params = {}

            first_row = session.query(ExperimentTableModel).first()
            if first_row is not None:
                x_axis_name = getattr(first_row, 'vote_metric', None)
                score_strategy_name = getattr(first_row, 'score_strategy', None)
            else:
                x_axis_name = x_axis.replace('_', ' ')

            query = session.query(ExperimentTableModel).filter_by(vote_generator_type=vote_generator_type,
                                                                  vote_distribution=vote_distrib)

            query = query.filter(ExperimentTableModel.metric_value >= vote_value_cuttoff)

            for param, value in fixed_params.items():
                query = query.filter(getattr(ExperimentTableModel, param) == value)

            if num_afs is not None:
                subquery = query.with_entities(
                    ExperimentTableModel.graph_type,
                    ExperimentTableModel.probability_value,
                    ExperimentTableModel.num_arguments,
                    func.group_concat(ExperimentTableModel.graph_name).label('graph_names')
                ).group_by(
                    ExperimentTableModel.graph_type,
                    ExperimentTableModel.probability_value,
                    ExperimentTableModel.num_arguments
                ).subquery()

                selected_graphs = {}
                for row in session.query(subquery).all():
                    key = (row.graph_type, row.probability_value, row.num_arguments)
                    graph_names = list(set(row.graph_names.split(',')))
                    selected_graphs[key] = random.sample(graph_names, min(num_afs, len(graph_names)))


                filtered_query = []
                for key, graph_names in selected_graphs.items():
                    graph_type, probability_value, num_arguments = key
                    filtered_query.append(
                        query.filter(
                            ExperimentTableModel.graph_type == graph_type,
                            ExperimentTableModel.probability_value == probability_value,
                            ExperimentTableModel.num_arguments == num_arguments,
                            ExperimentTableModel.graph_name.in_(graph_names)
                        )
                    )

                query = filtered_query[0].union_all(*filtered_query[1:])


            first_row = query.first()
            if first_row is not None:
                score_strategy_name = getattr(first_row, 'score_strategy', None)

            results = [
                {
                    'x_value': getattr(r, x_axis),
                    'y_value': getattr(r, y_axis),
                    'aggregation_name': r.aggregation_name,
                    'agg_metric_1': r.agg_metric_1,
                    'agg_metric_2': r.agg_metric_2
                }
                for r in query
            ]

            filtered_results = [
                r for r in results
                if not any(keyword == r['aggregation_name'] or
                           keyword == r['agg_metric_1'] or
                           keyword == r['agg_metric_2']
                           for keyword in exclude_keywords)
            ]

            if len(filtered_results) == 0:
                session.close()
                return

            grouped = {}
            for r in filtered_results:
                key = (r['x_value'], r['aggregation_name'], r['agg_metric_1'], r['agg_metric_2'])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(r['y_value'])

            mean_values = {
                key: sum(values) / len(values) for key, values in grouped.items()
            }

            curves = {}
            for (x_value, aggregation, agg_metric_1, agg_metric_2), y_value in mean_values.items():
                if aggregation not in ["JA", "TestingSemantic", "ARS", "TODF"]:
                    label = f"{aggregation}: {agg_metric_1} & {agg_metric_2}"
                else:
                    if aggregation == "JA":
                        label = f"LA: {agg_metric_2}"
                    elif aggregation == "TODF":
                        label = f"{aggregation}: {agg_metric_2}"
                    elif aggregation == "TestingSemantic":
                        label = "All extensions"
                    elif aggregation == "ARS":
                        label = f"{aggregation}: {agg_metric_2}"
                if label not in curves:
                    curves[label] = []
                curves[label].append((x_value,y_value))

            style_mapping = final_style_mapping

            default_style = {"color": "black", "linestyle": ":"}

            plt.figure(figsize=(10, 8))

            for label, points in curves.items():
                points = sorted(points, key=lambda p: p[0])
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]

                style = style_mapping.get(label, default_style)

                plt.plot(x_values, y_values, label=label, color=style["color"], linestyle=style["linestyle"],
                         linewidth=4, alpha=0.7)

            plt.xlabel(x_axis_name.title(), fontsize=16)
            plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=16)
            if title is None:
                title = f'{y_axis.replace("_", " ").title()} vs. {x_axis.replace("_", " ").title()} by Aggregation Method'
            plt.title(title, fontsize=16)

            if "Absaf" in score_strategy_name:
                plt.ylim(-0.03, 1.03)
            else:
                plt.ylim(-3, 103)

            plt.legend(title='Aggregation Method & Operators', loc='lower right', fontsize=12.5, title_fontsize=14)

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            if invert_x:
                plt.gca().invert_xaxis()
            plt.show()

        finally:
            session.close()

    def plot_score_with_confidence(self, x_axis, y_axis, vote_generator_type, vote_distrib, num_afs,
                                   exclude_keywords=None,
                                   fixed_params=None, vote_value_cuttoff=0, title=None, invert_x=False, ci=95):
        session = self.Session()

        try:
            if exclude_keywords is None:
                exclude_keywords = []
            if fixed_params is None:
                fixed_params = {}

            first_row = session.query(ExperimentTableModel).first()
            if first_row is not None:
                x_axis_name = getattr(first_row, 'vote_metric', None)
                score_strategy_name = getattr(first_row, 'score_strategy', None)
            else:
                x_axis_name = x_axis.replace('_', ' ')

            query = session.query(ExperimentTableModel).filter_by(vote_generator_type=vote_generator_type,
                                                                  vote_distribution=vote_distrib)

            query = query.filter(ExperimentTableModel.metric_value >= vote_value_cuttoff)

            for param, value in fixed_params.items():
                query = query.filter(getattr(ExperimentTableModel, param) == value)

            if num_afs is not None:
                subquery = query.with_entities(
                    ExperimentTableModel.graph_type,
                    ExperimentTableModel.probability_value,
                    ExperimentTableModel.num_arguments,
                    func.group_concat(ExperimentTableModel.graph_name).label('graph_names')
                ).group_by(
                    ExperimentTableModel.graph_type,
                    ExperimentTableModel.probability_value,
                    ExperimentTableModel.num_arguments
                ).subquery()

                selected_graphs = {}
                for row in session.query(subquery).all():
                    key = (row.graph_type, row.probability_value, row.num_arguments)
                    graph_names = list(set(row.graph_names.split(',')))
                    selected_graphs[key] = random.sample(graph_names, min(num_afs, len(graph_names)))

                filtered_query = []
                for key, graph_names in selected_graphs.items():
                    graph_type, probability_value, num_arguments = key
                    filtered_query.append(
                        query.filter(
                            ExperimentTableModel.graph_type == graph_type,
                            ExperimentTableModel.probability_value == probability_value,
                            ExperimentTableModel.num_arguments == num_arguments,
                            ExperimentTableModel.graph_name.in_(graph_names)
                        )
                    )

                if filtered_query:
                    query = filtered_query[0].union_all(*filtered_query[1:])

            first_row = query.first()
            if first_row is not None:
                score_strategy_name = getattr(first_row, 'score_strategy', None)

            results = [
                {
                    'x_value': getattr(r, x_axis),
                    'y_value': getattr(r, y_axis),
                    'aggregation_name': r.aggregation_name,
                    'agg_metric_1': r.agg_metric_1,
                    'agg_metric_2': r.agg_metric_2
                }
                for r in query
            ]

            filtered_results = [
                r for r in results
                if not any(keyword == r['aggregation_name'] or
                           keyword == r['agg_metric_1'] or
                           keyword == r['agg_metric_2']
                           for keyword in exclude_keywords)
            ]

            if len(filtered_results) == 0:
                session.close()
                return

            for r in filtered_results:
                if r['aggregation_name'] not in ["JA", "AllExtensionsBaseline", "ARS", "MR"]:
                    r['label'] = f"{r['aggregation_name']}: {r['agg_metric_1']} & {r['agg_metric_2']}"
                else:
                    if r['aggregation_name'] == "JA":
                        r['label'] = f"LA: {r['agg_metric_2']}"
                    elif r['aggregation_name'] == "MR":
                        r['label'] = f"Majority Rule"
                    elif r['aggregation_name'] == "AllExtensionsBaseline":
                        r['label'] = "All extensions"
                    elif r['aggregation_name'] == "ARS":
                        r['label'] = f"{r['aggregation_name']}"

            df = pd.DataFrame(filtered_results)

            style_mapping = final_style_mapping

            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 8))

            unique_labels = df['label'].unique()

            for label in unique_labels:
                label_df = df[df['label'] == label]

                style = style_mapping.get(label, {"color": "black", "linestyle": ":", "marker": "o"})

                sns.lineplot(
                    data=label_df,
                    x='x_value',
                    y='y_value',
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    # linestyle=":",
                    marker=style["marker"],
                    markevery=1,
                    markersize=16,
                    linewidth=3,
                    errorbar=("ci", ci),
                    err_style='band'
                )

            plt.xlabel(x_axis_name.title() if x_axis_name else x_axis.replace('_', ' ').title(), fontsize=20)
            if y_axis == "score" and score_strategy_name == "SimpleScore":
                plt.ylabel("AM " + y_axis.replace('_', ' ').title(), fontsize=20)
            elif y_axis == "score" and score_strategy_name == "SimilarityScore":
                plt.ylabel("SM " + y_axis.replace('_', ' ').title(), fontsize=20)
            else:
                plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=20)
            if title is None:
                title = f'{y_axis.replace("_", " ").title()} vs. {x_axis.replace("_", " ").title()} by Aggregation Method'
            plt.title(title, fontsize=16)

            if score_strategy_name and "Absaf" in score_strategy_name:
                plt.ylim(-0.03, 1.03)
                plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16)
            else:
                plt.ylim(-3, 103)
                plt.yticks(np.arange(0, 101, 10), fontsize=16)
            plt.xticks(fontsize=16)
            plt.legend(title='', loc='lower right', fontsize=18, title_fontsize=14)

            plt.tight_layout()

            if invert_x:
                plt.gca().invert_xaxis()

            plt.show()

        finally:
            session.close()

    def plot_score_dispersion(self, x_axis, y_axis, vote_generator_type='Our_vote', exclude_keywords=None):
        """Plot score dispersion across different aggregation methods and parameters"""
        session = self.Session()

        try:
            if exclude_keywords is None:
                exclude_keywords = []

            query = session.query(ExperimentTableModel).filter_by(
                vote_generator_type=vote_generator_type
            )

            results = [
                {
                    'vote_metric': r.metric_value,
                    'aggregation_name': r.aggregation_name,
                    'agg_metric_1': r.agg_metric_1,
                    'agg_metric_2': r.agg_metric_2,
                    'score': r.score
                }
                for r in query
            ]

            filtered_results = [
                r for r in results
                if not any(keyword.lower() in (r['aggregation_name'] + r['agg_metric_1'] + r['agg_metric_2']).lower()
                           for keyword in exclude_keywords)
            ]

            grouped = {}
            for r in filtered_results:
                key = (r['vote_metric'], r['aggregation_name'], r['agg_metric_1'], r['agg_metric_2'])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(r['score'])

            mean_scores = {
                key: sum(scores) / len(scores) for key, scores in grouped.items()
            }

            unique_x_values = sorted(set(k[0] for k in mean_scores.keys()))
            curves = {}

            for (voter_reliability, aggregation, agg_metric_1, agg_metric_2), score in mean_scores.items():
                label = f"{aggregation}: {agg_metric_1} & {agg_metric_2}"
                if label not in curves:
                    curves[label] = []
                curves[label].append((voter_reliability, score))

            plt.figure(figsize=(12, 8))

            for label, points in curves.items():
                points = sorted(points, key=lambda p: p[0])
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                plt.plot(x_values, y_values, label=label)

            plt.xlabel(x_axis.capitalize())
            plt.ylabel(y_axis.capitalize())
            plt.title(f'{y_axis.capitalize()} vs. {x_axis.capitalize()} by Aggregation Method')

            plt.legend(title='Aggregation Method & Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid()
            plt.tight_layout()

            plt.show()

        finally:
            session.close()

    def plot_results(self, parameter_x, parameter_y, **query_filters):
        """Plot results based on two parameters using the query_votes_raw method"""
        results = self.query_votes_raw(**query_filters)

        x_values = [getattr(r, parameter_x) for r in results]
        y_values = [getattr(r, parameter_y) for r in results]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, alpha=0.7)
        plt.xlabel(parameter_x)
        plt.ylabel(parameter_y)
        plt.title(f'{parameter_y} vs. {parameter_x}')
        plt.grid(True)
        plt.show()

    @contextmanager
    def session_scope(self,session_factory):
        """Context manager for database sessions"""
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def merge_dataset(self, other_db_path, batch_size=10000, limit=None, offset=0, check_unique=True):
        other_engine = create_engine(f'sqlite:///{other_db_path}')
        OtherSession = sessionmaker(bind=other_engine)

        total_merged = 0
        total_duplicates = 0

        try:
            columns = [column.name for column in ExperimentTableModel.__table__.columns if column.name != 'id']
            columns_str = ', '.join(columns)

            unique_columns = [
                'graph_type', 'num_arguments', 'graph_generation_metrics', 'graph_name', 'semantic',
                'vote_generator_type', 'vote_type', 'vote_distribution',
                'vote_metric', 'metric_value', 'aggregation_name',
                'agg_metric_1', 'agg_metric_2', 'num_votes',
                'score_strategy', 'score_metric'
            ]

            unique_condition = ' AND '.join([f"target.{col} = source.{col}" for col in unique_columns])

            with self.session_scope(OtherSession) as other_session, \
                    self.session_scope(self.Session) as current_session:
                if limit is None:
                    total_count = other_session.query(ExperimentTableModel).count()
                    limit = total_count - offset

                for batch_offset in range(offset, offset + limit, batch_size):
                    batch_limit = min(batch_size, limit - (batch_offset - offset))

                    batch_query = other_session.query(ExperimentTableModel) \
                        .limit(batch_limit).offset(batch_offset)
                    batch_data = [
                        {column: getattr(row, column) for column in columns}
                        for row in batch_query.all()
                    ]

                    if not batch_data:
                        break

                    if check_unique:
                        create_temp_table_sql = f"""
                                CREATE TEMPORARY TABLE temp_batch_data (
                                    {', '.join(f'{col} {ExperimentTableModel.__table__.columns[col].type}'
                                               for col in columns)}
                                )
                            """
                        current_session.execute(text(create_temp_table_sql))

                        insert_temp_sql = f"""
                                INSERT INTO temp_batch_data ({columns_str})
                                VALUES ({', '.join([':' + col for col in columns])})
                            """
                        insert_temp_stmt = text(insert_temp_sql)
                        for row in batch_data:
                            current_session.execute(insert_temp_stmt, row)

                        insert_sql = f"""
                                INSERT INTO {ExperimentTableModel.__tablename__} ({columns_str})
                                SELECT source.*
                                FROM temp_batch_data source
                                WHERE NOT EXISTS (
                                    SELECT 1
                                    FROM {ExperimentTableModel.__tablename__} target
                                    WHERE {unique_condition}
                                )
                            """
                        result = current_session.execute(text(insert_sql))

                        current_session.execute(text("DROP TABLE temp_batch_data"))

                        rows_inserted = result.rowcount
                        total_merged += rows_inserted
                        total_duplicates += (len(batch_data) - rows_inserted)

                    else:
                        insert_sql = f"""
                                INSERT INTO {ExperimentTableModel.__tablename__} ({columns_str})
                                VALUES ({', '.join([':' + col for col in columns])})
                            """
                        insert_stmt = text(insert_sql)
                        for row in batch_data:
                            current_session.execute(insert_stmt, row)
                        total_merged += len(batch_data)

                    current_session.commit()

            return total_merged, total_duplicates

        except SQLAlchemyError as e:
            raise RuntimeError(f"Error during database merge: {e}")

    def plot_extensions_by_arguments(self):
        """
        Plot a bar chart where the y-axis is the number of extensions
        the x-axis is ticks of the number of arguments, and the bar colors represent graph types
        """
        query = self.session.query(
            ExperimentTableModel.num_arguments,
            ExperimentTableModel.num_extensions,
            ExperimentTableModel.graph_type
        ).all()

        if not query:
            print("No data found in the database.")
            return

        data = [
            {'num_arguments': result.num_arguments,
             'num_extensions': result.num_extensions,
             'graph_type': result.graph_type}
            for result in query
        ]

        import pandas as pd
        df = pd.DataFrame(data)

        sns.set_palette("Set2")

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=df,
            x="num_arguments",
            y="num_extensions",
            hue="graph_type",
            dodge=True
        )

        plt.xlabel("Number of Arguments", fontsize=12)
        plt.ylabel("Number of Extensions", fontsize=12)
        plt.title("Number of Extensions vs. Number of Arguments by Graph Type", fontsize=14)
        plt.legend(title="Graph Type", fontsize=10, title_fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_score_by_parameter(self,score_strategy,
               x_axis='metric_value',
               y_axis='score',
               aggregation_name=None,
               agg_metric_1=None,
               agg_metric_2=None,
               color_param='num_votes',
               vote_generator_type='Our_vote',
               exclude_keywords=None,
               fixed_params=None,
               vote_value_cuttoff=40,
               title = None):
        session = self.Session()
        try:
            if exclude_keywords is None:
                exclude_keywords = []
            if fixed_params is None:
                fixed_params = {}

            query = session.query(ExperimentTableModel).filter_by(vote_generator_type=vote_generator_type,
                                                                  score_strategy=score_strategy)

            if aggregation_name:
                query = query.filter(ExperimentTableModel.aggregation_name == aggregation_name)
            if agg_metric_1:
                query = query.filter(ExperimentTableModel.agg_metric_1 == agg_metric_1)
            if agg_metric_2:
                query = query.filter(ExperimentTableModel.agg_metric_2 == agg_metric_2)

            query = query.filter(ExperimentTableModel.metric_value >= vote_value_cuttoff)

            for param, value in fixed_params.items():
                if value is not None:
                    query = query.filter(getattr(ExperimentTableModel, param) == value)

            results = [
                {
                    'x_value': getattr(r, x_axis),
                    'y_value': getattr(r, y_axis),
                    'color_value': getattr(r, color_param),
                    'aggregation_name': r.aggregation_name,
                    'agg_metric_1': r.agg_metric_1,
                    'agg_metric_2': r.agg_metric_2
                }
                for r in query
            ]
            if len(results) == 0:
                return

            filtered_results = [
                r for r in results
                if not any(keyword == r['aggregation_name'] or
                           keyword == r['agg_metric_1'] or
                           keyword == r['agg_metric_2']
                           for keyword in exclude_keywords)
            ]

            plt.figure(figsize=(15, 10))

            color_values = sorted(set(r['color_value'] for r in filtered_results))
            color_map = plt.cm.viridis(np.linspace(0, 1, len(color_values)))
            color_dict = dict(zip(color_values, color_map))

            used_color_values = set()

            for color_value in color_values:
                color_results = [r for r in filtered_results if r['color_value'] == color_value]

                grouped = {}
                for r in color_results:
                    if r['x_value'] not in grouped:
                        grouped[r['x_value']] = []
                    grouped[r['x_value']].append(r['y_value'])

                mean_values = {
                    x_value: sum(values) / len(values) for x_value, values in grouped.items()
                }

                sorted_points = sorted(mean_values.items())

                x_values = [point[0] for point in sorted_points]
                y_values = [point[1] for point in sorted_points]

                label = f"{color_param}={color_value}" if color_value not in used_color_values else None

                plt.plot(x_values, y_values,
                         marker='o',
                         color=color_dict[color_value],
                         label=label)

                if color_value not in used_color_values:
                    used_color_values.add(color_value)

            plt.xlabel(x_axis.replace('_', ' ').title())
            plt.ylabel(y_axis.replace('_', ' ').title())
            if title is None:
                title = (f'{y_axis.replace("_", " ").title()} vs. {x_axis.replace("_", " ").title()}'
                         f'\nAggregation: {aggregation_name or "All"}, '
                         f'Color by {color_param}')
            plt.title(title)

            plt.legend(title=color_param.replace('_', ' ').title(),
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            plt.show()

        finally:
            session.close()

    def add_new_column(self, table_name, column_name, column_value):
        """
        Add num_extensions column to the existing votes table

        This method alters the existing database schema to include
        the new num_extensions column with a default value.
        """
        from sqlalchemy.sql import text

        session = self.Session()
        try:
            # Add the new column with a default value of 0
            session.execute(text(f"""
                ALTER TABLE {table_name} ADD COLUMN {column_name} INTEGER DEFAULT {column_value}
            """))
            session.commit()
            print("Successfully added num_extensions column to votes table")
        except Exception as e:
            session.rollback()
            print(f"Error adding num_extensions column: {e}")
        finally:
            session.close()

    def plot_score_by_parameter2(self, num_bins, bin_mode, score_strategy, x_axis='metric_value', y_axis='score', metric_value=0.6,
                                 vote_generator_type='Our_vote', exclude_keywords=None, fixed_params=None,
                                 vote_value_cuttoff=40, title=None, enable_binning=True):
        """Plot scores against parameters with optional binning for handling many unique x values"""
        session = self.Session()
        try:
            style_mapping = final_style_mapping

            if exclude_keywords is None:
                exclude_keywords = []
            if fixed_params is None:
                fixed_params = {}

            query = session.query(ExperimentTableModel).filter_by(
                vote_generator_type=vote_generator_type,
                score_strategy=score_strategy
            )

            query = query.filter(ExperimentTableModel.metric_value == metric_value)

            for param, value in fixed_params.items():
                if value is not None:
                    query = query.filter(getattr(ExperimentTableModel, param) == value)

            results = [
                {
                    'x_value': getattr(r, x_axis),
                    'y_value': getattr(r, y_axis),
                    'aggregation_name': r.aggregation_name,
                    'agg_metric_1': r.agg_metric_1,
                    'agg_metric_2': r.agg_metric_2
                }
                for r in query
            ]
            if len(results) == 0:
                return

            for r in results:
                if r['aggregation_name'] not in ["JA", "AllExtensionsBaseline", "ARS", "MR"]:
                    r['color_value'] = f"{r['aggregation_name']}: {r['agg_metric_1']} & {r['agg_metric_2']}"
                else:
                    if r['aggregation_name'] == "JA":
                        r['color_value'] = f"LA: {r['agg_metric_2']}"
                    elif r['aggregation_name'] == "MR":
                        r['color_value'] = f"Majority Rule"
                    elif r['aggregation_name'] == "AllExtensionsBaseline":
                        r['color_value'] = "All extensions"
                    elif r['aggregation_name'] == "ARS":
                        r['color_value'] = f"{r['aggregation_name']}"

            filtered_results = [
                r for r in results
                if not any(keyword == r['aggregation_name'] or
                           keyword == r['agg_metric_1'] or
                           keyword == r['agg_metric_2']
                           for keyword in exclude_keywords)
            ]

            unique_x_values = sorted(set(r['x_value'] for r in filtered_results))

            if enable_binning and len(unique_x_values) > 20:
                return self._plot_with_binning(filtered_results=filtered_results,
                                               x_axis=x_axis, y_axis=y_axis,
                                               title=title, num_bins=num_bins,
                                               bin_mode=bin_mode, style_mapping=style_mapping,
                                               score_strategy=score_strategy)
            else:
                plt.figure(figsize=(10, 8))

                for color_value in sorted(set(r['color_value'] for r in filtered_results)):
                    color_results = [r for r in filtered_results if r['color_value'] == color_value]
                    data = pd.DataFrame(color_results)
                    style = style_mapping.get(color_value, {"color": "black", "linestyle": ":", "marker": "o"})
                    sns.lineplot(
                        data=data,
                        x='x_value',
                        y='y_value',
                        label=color_value,
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        markevery=1,
                        markersize=14,
                        linewidth=3,
                        ci=95
                    )

                plt.xlabel(x_axis.replace('num', 'Number of').replace('_', ' ').title(), fontsize=20)
                if y_axis == "score" and score_strategy == "SimpleScore":
                    plt.ylabel("AM " + y_axis.replace('_', ' ').title(), fontsize=20)
                elif y_axis == "score" and score_strategy == "SimilarityScore":
                    plt.ylabel("SM " + y_axis.replace('_', ' ').title(), fontsize=20)
                else:
                    plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=20)
                if title is None:
                    title = (f'{y_axis.replace("_", " ").title()} vs. {x_axis.replace("_", " ").title()}'
                             f'\nColor by Aggregation (Combination of Name and Metrics)')
                plt.title(title, fontsize=16)

                plt.legend([], [], frameon=False)
                # plt.legend(title='', loc='lower right', fontsize=18, title_fontsize=14)
                plt.ylim(-3, 103)
                plt.yticks(np.arange(0, 101, 10), fontsize=16)
                plt.xticks(fontsize=16)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()

        finally:
            session.close()

    def _plot_with_binning(self, filtered_results, x_axis, y_axis, title, num_bins, score_strategy,
                           bin_mode='equal_width',
                           style_mapping=None):
        """Helper method to plot data with binning for x-axis values, with evenly spaced bins on x-axis"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.DataFrame(filtered_results)

        min_x = df['x_value'].min()
        max_x = df['x_value'].max()

        if bin_mode == 'equal_width':
            bins = np.linspace(min_x, max_x, num_bins + 1)
        elif bin_mode == 'equal_count':
            bins = np.sort(list(set(np.percentile(df['x_value'], np.linspace(0, 100, num_bins + 1)))), axis=None)
        elif bin_mode == 'custom':
            if min_x <= 0:
                min_x = 0.1
            bins = np.logspace(np.log10(min_x), np.log10(max_x), num_bins + 1)
        else:
            raise ValueError(f"Invalid bin_mode: {bin_mode}")

        bin_labels = [f"[{bins[i]:.0f}, {bins[i + 1]:.0f}]" for i in range(len(bins) - 1)]

        df['bin_index'] = pd.cut(df['x_value'], bins=bins, labels=range(len(bins) - 1), include_lowest=True)

        df['bin_position'] = df['bin_index'].astype(float)

        plt.figure(figsize=(10, 8))

        for color_value in sorted(df['color_value'].unique()):
            color_df = df[df['color_value'] == color_value]
            style = style_mapping.get(color_value, {"color": "black", "linestyle": ":", "marker": "o"})

            sns.lineplot(
                data=color_df,
                x='bin_position',
                y='y_value',
                label=color_value,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markevery=1,
                markersize=14,
                linewidth=3,
                ci=95
            )

        plt.xticks(range(len(bins) - 1), labels=bin_labels, fontsize=14)  # , rotation=45)

        plt.xlim(-0.1, len(bins) - 1.8)

        plt.xlabel(x_axis.replace('num', 'Number of').replace('_', ' ').title(), fontsize=18)
        if y_axis == "score" and score_strategy == "SimpleScore":
            plt.ylabel("AM " + y_axis.replace('_', ' ').title(), fontsize=20)
        elif y_axis == "score" and score_strategy == "SimilarityScore":
            plt.ylabel("SM " + y_axis.replace('_', ' ').title(), fontsize=20)
        else:
            plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=20)

        if title is None:
            title = (f'{y_axis.replace("_", " ").title()} vs. {x_axis.replace("_", " ").title()} (Binned)'
                     f'\nBin Mode: {bin_mode}, Number of Bins: {num_bins}, Evenly Spaced Display')
        plt.title(title, fontsize=16)

        plt.legend([], [], frameon=False)
        plt.ylim(-3, 103)
        plt.yticks(np.arange(0, 101, 10), fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()

    def plot_by_graph_type(self, x_axis, y_axis, num_votes,
                            vote_generator_type, vote_distrib,
                           aggregation_name_1, agg_metric_1_1,
                            agg_metric_2_1, semantic,
                           aggregation_name_2, agg_metric_1_2,
                            agg_metric_2_2,
                           vote_value_cuttoff=40,
                            title=None, invert_x=False, ci=95):
        session = self.Session()

        try:
            first_row = session.query(ExperimentTableModel).first()
            if first_row is not None:
                x_axis_name = getattr(first_row, 'vote_metric', None)
                score_strategy_name = getattr(first_row, 'score_strategy', None)
            else:
                x_axis_name = x_axis.replace('_', ' ')

            query = session.query(ExperimentTableModel).filter_by(vote_generator_type=vote_generator_type,
                                                                  vote_distribution=vote_distrib,
                                                                  semantic=semantic,
                                                                  num_votes=num_votes)

            query = query.filter(ExperimentTableModel.metric_value >= vote_value_cuttoff)

            condition1 = (
                    (ExperimentTableModel.aggregation_name == aggregation_name_1) &
                    (ExperimentTableModel.agg_metric_1 == agg_metric_1_1) &
                    (ExperimentTableModel.agg_metric_2 == agg_metric_2_1)
            )

            condition2 = (
                    (ExperimentTableModel.aggregation_name == aggregation_name_2) &
                    (ExperimentTableModel.agg_metric_1 == agg_metric_1_2) &
                    (ExperimentTableModel.agg_metric_2 == agg_metric_2_2)
            )

            condition3 = (
                    (ExperimentTableModel.aggregation_name == "ABSAF") &
                    (ExperimentTableModel.agg_metric_1 == "all") &
                    (ExperimentTableModel.agg_metric_2 == "harmonic")
            )

            color_1 = dark_color  # "#1A1AFF"  # First aggregation
            color_2 = medium_color  # "#AB7DB6"  # Second aggregation
            color_3 = medium_light_color  # "#ad895a"  # Second aggregation

            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 8))

            graph_types = ["BarabasiAlbert", "Gilbert", "WattsStrogatz"]

            marker_mapping = {
                graph_type: marker for graph_type, marker in
                zip(graph_types, ["P", "D", "p", "h"])
            }

            for condition, color in [(condition1, color_1), (condition2, color_2), (condition3, color_3)]:
                for graph_type in graph_types:
                    current_query = query.filter(condition).filter(ExperimentTableModel.graph_type==graph_type)

                    results = [
                        {
                            'x_value': getattr(r, x_axis),
                            'y_value': getattr(r, y_axis),
                            'aggregation_name': r.aggregation_name,
                            'agg_metric_1': r.agg_metric_1,
                            'agg_metric_2': r.agg_metric_2,
                            'graph_type': r.graph_type
                        }
                        for r in current_query
                    ]

                    if len(results) == 0:
                        print("No results found matching the criteria.")
                        session.close()
                        return

                    for r in results:
                        if r['aggregation_name'] not in ["JA", "AllExtensionsBaseline", "ARS", "MR"]:
                            r['agg_label'] = f"{r['aggregation_name']}: {r['agg_metric_1']} & {r['agg_metric_2']}"
                        else:
                            if r['aggregation_name'] == "JA":
                                r['agg_label'] = f"LA: {r['agg_metric_2']}"
                            elif r['aggregation_name'] == "MR":
                                r['agg_label'] = f"Majority Rule"
                            elif r['aggregation_name'] == "AllExtensionsBaseline":
                                r['agg_label'] = "All extensions"
                            elif r['aggregation_name'] == "ARS":
                                r['agg_label'] = f"{r['aggregation_name']}"

                        r['label'] = f"{r['agg_label']} - {r['graph_type']}"
                    if r['graph_type'] == "Gilbert":
                        label = f"{r['agg_label']} - ER"
                    elif r['graph_type'] == "BarabasiAlbert":
                        label = f"{r['agg_label']} - BA"
                    elif r['graph_type'] == "WattsStrogatz":
                        label = f"{r['agg_label']} - WS"

                    # Convert to DataFrame for seaborn
                    df = pd.DataFrame(results)

                    # Get marker for this graph_type
                    marker = marker_mapping.get(graph_type, "o")

                    linestyle = "-"

                    subset_df = df[(df['graph_type'] == graph_type)]

                    sns.lineplot(
                        data=subset_df,
                        x='x_value',
                        y='y_value',
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markevery=1,
                        markersize=16,
                        linewidth=3,
                        ci=ci
                    )

            plt.xlabel(x_axis_name.title() if x_axis_name else x_axis.replace('_', ' ').title(), fontsize=20)
            if y_axis == "score" and score_strategy_name == "SimpleScore":
                plt.ylabel("AM " + y_axis.replace('_', ' ').title(), fontsize=20)
            elif y_axis == "score" and score_strategy_name == "SimilarityScore":
                plt.ylabel("SM " + y_axis.replace('_', ' ').title(), fontsize=20)
            else:
                plt.ylabel(y_axis.replace('_', ' ').title(), fontsize=20)
            if title is not None:
                plt.title(title, fontsize=20)

            if score_strategy_name and "Absaf" in score_strategy_name:
                plt.ylim(-0.03, 1.03)
                plt.yticks(np.arange(0, 1.1, 0.1), fontsize=16)
            else:
                plt.ylim(-3, 103)
                plt.yticks(np.arange(0, 101, 10), fontsize=16)
            plt.xticks(fontsize=14)

            plt.legend(title='', loc='lower right', fontsize=18, title_fontsize=14)

            plt.tight_layout()

            if invert_x:
                plt.gca().invert_xaxis()

            plt.show()

        finally:
            session.close()
