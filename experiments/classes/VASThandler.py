# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Tuple, Dict, Any
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import json

from project_config import FrameworksPath
from experiments.config import data_structure, vote_params
from structure.basic_vote_strategy import BasicStrategy
from structure.ABSAF_vote_strategy import ABSAFVoteStrategy
from structure.opinion_based_af import OBAF
from structure.extension import Extension
from afbenchgen2.generate_afs_from_java import AFBenchGraphGenerator

from itertools import product


Base = declarative_base()


class GraphModel(Base):
    """Database model representing an Argumentation Framework (Graph)"""
    __tablename__ = 'graphs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_type = Column(String, nullable=False)
    num_arguments = Column(Integer, nullable=False)
    graph_generation_metrics = Column(JSON, nullable=False)
    graph_name = Column(Integer, nullable=False)
    graph_apx = Column(Text, nullable=False)

    extensions = Column(JSON, nullable=False)
    # extensions = Column(MutableDict.as_mutable(JSONB))
    pref_num_extensions = Column(Integer, nullable=False)
    comp_num_extensions = Column(Integer, nullable=False)
    pref_ground_truth = Column(JSON, nullable=False)
    comp_ground_truth = Column(JSON, nullable=False)

    votes = relationship("VoteModel", back_populates="graph")
    absaf_votes = relationship("ABSAFVoteModel", back_populates="graph")


class VoteModel(Base):
    """Database model representing a set of votes generated for a specific graph"""
    __tablename__ = 'votes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(Integer, ForeignKey('graphs.id'), nullable=False)

    semantic = Column(String, nullable=False)
    vote_generator_type = Column(String, nullable=False)
    vote_type = Column(String, nullable=False)
    votes_distribution = Column(String, nullable=False)
    vote_metric = Column(String, nullable=False)
    vote_metric_value = Column(Float, nullable=False)
    votes = Column(Text, nullable=False)

    graph = relationship("GraphModel", back_populates="votes")

class ABSAFVoteModel(Base):
    """Database model representing ABSAF specific votes"""
    __tablename__ = 'absaf_votes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(Integer, ForeignKey('graphs.id'), nullable=False)

    semantic = Column(String, nullable=False)
    vote_generator_type = Column(String, nullable=False)
    vote_type = Column(String, nullable=False)
    votes_distribution = Column(String, nullable=False)
    vote_metric = Column(String, nullable=False)
    vote_metric_value = Column(Float, nullable=False)
    votes = Column(Text, nullable=False)

    graph = relationship("GraphModel", back_populates="absaf_votes")


_absaf_votes = "ABSAF_vote"
_our_votes = "Our_vote"
_num_votes = 100


class VASTHandler:
    """
    Handles all interactions with the SQLite database, including graph generation,
    storage, retrieval, and vote generation.
    """
    def __init__(self, database_name: str):
        self.db_path = database_name
        exists = os.path.exists(self.db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        print("Create database")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        if exists:
            print("Loading database")
            # Load the existing database
            print(f"Database at {self.db_path} found. Loading...")
            self.load_database(self.db_path)

        self.vote_distributions = ["mean"]
        self.vote_types = ["nozeros", "zeros"]

    def insert_graph(self,
                     graph_type: str,
                     num_arguments: int,
                     probability_type: str,
                     probability_value: float,
                     graph_name: int,
                     graph_apx: str) -> int:
        """Insert a new graph into the database, avoiding duplicates"""
        if graph_type not in ['BarabasiAlbert', 'ErdosRenyi']:
            raise ValueError(f"Invalid graph type: {graph_type}")

        if not (5 <= num_arguments <= 40):
            raise ValueError(f"Number of arguments must be between 5 and 40, got {num_arguments}")

        if not (0 <= probability_value <= 1):
            raise ValueError(f"Probability must be between 0 and 1, got {probability_value}")

        if not (1 <= graph_name <= 100):
            raise ValueError(f"Graph name must be between 0 and 1, got {graph_name}")

        session = self.Session()
        try:
            existing_graph = session.query(GraphModel).filter_by(
                graph_type=graph_type,
                num_arguments=num_arguments,
                probability_type=probability_type,
                probability_value=probability_value,
                graph_name=graph_name
            ).first()

            if existing_graph:
                print("Graph exists")
                return existing_graph.id

            graph = GraphModel(
                graph_type=graph_type,
                num_arguments=num_arguments,
                probability_type=probability_type,
                probability_value=probability_value,
                graph_name=graph_name,
                graph_apx=graph_apx
            )
            session.add(graph)
            session.commit()
            graph_id = graph.id

        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError("Failed to insert graph") from e
        finally:
            session.close()

        return graph_id

    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm.exc import NoResultFound

    def insert_vote(self,
                    graph_type: str,
                    num_arguments: int,
                    probability_type: str,
                    probability_value: float,
                    graph_name: int,
                    semantic: str,
                    num_extensions: int,
                    extension: str,
                    vote_generator_type: str,
                    vote_type: str,
                    votes_distribution: str,
                    vote_metric: str,
                    metric_value: float,
                    votes: str) -> int:
        """Insert a new vote into the database, avoiding duplicates"""

        if vote_generator_type not in [_absaf_votes, _our_votes]:
            raise ValueError(f"Invalid vote generator type: {vote_generator_type}")

        if vote_metric not in ['dispersion', 'reliability']:
            raise ValueError(f"Invalid vote metric: {vote_metric}")

        session = self.Session()
        try:
            graph = session.query(GraphModel).filter_by(
                graph_type=graph_type,
                num_arguments=num_arguments,
                probability_type=probability_type,
                probability_value=probability_value,
                graph_name=graph_name
            ).one_or_none()

            if not graph:
                raise ValueError("Graph not found. You must insert the graph first before inserting its vote.")

            existing_vote = session.query(VoteModel).filter_by(
                graph_id=graph.id,
                semantic=semantic,
                num_extensions=num_extensions,
                ground_truth=extension,
                vote_generator_type=vote_generator_type,
                vote_type=vote_type,
                votes_distribution=votes_distribution,
                vote_metric=vote_metric,
                vote_metric_value=metric_value,
                votes=votes
            ).first()

            if existing_vote:
                return existing_vote.id

            new_vote = VoteModel(
                graph_id=graph.id,
                semantic=semantic,
                num_extensions=num_extensions,
                ground_truth=extension,
                vote_generator_type=vote_generator_type,
                vote_type=vote_type,
                votes_distribution=votes_distribution,
                vote_metric=vote_metric,
                vote_metric_value=metric_value,
                votes=votes
            )
            session.add(new_vote)
            session.commit()
            vote_id = new_vote.id

        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

        return vote_id

    def query_graphs(self, **kwargs) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            query = session.query(GraphModel)
            for key, value in kwargs.items():
                if not hasattr(GraphModel, key):
                    raise AttributeError(f"GraphModel has no attribute '{key}'")
                query = query.filter(getattr(GraphModel, key) == value)

            results = [
                {
                    'id': graph.id,
                    'graph_type': graph.graph_type,
                    'num_arguments': graph.num_arguments,
                    'graph_generation_metrics': graph.graph_generation_metrics,
                    'graph_name': graph.graph_name,
                    'graph_apx': graph.graph_apx,
                    'extensions': graph.extensions,
                    'pref_num_extensions': graph.pref_num_extensions,
                    'comp_num_extensions': graph.comp_num_extensions,
                    'pref_ground_truth': graph.pref_ground_truth,
                    'comp_ground_truth': graph.comp_ground_truth,
                } for graph in query.all()
            ]
            return results
        finally:
            session.close()

    def query_votes(self, **kwargs) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            query = session.query(VoteModel).join(GraphModel)

            for key, value in kwargs.items():
                if hasattr(VoteModel, key):
                    query = query.filter(getattr(VoteModel, key) == value)
                elif hasattr(GraphModel, key):
                    query = query.filter(getattr(GraphModel, key) == value)
                else:
                    raise AttributeError(f"Unknown attribute '{key}'")

            results = [
                {
                    'id': vote.id,
                    'graph_id': vote.graph_id,
                    'semantic': vote.semantic,
                    'vote_generator_type': vote.vote_generator_type,
                    'vote_type': vote.vote_type,
                    'votes_distribution': vote.votes_distribution,
                    'vote_metric': vote.vote_metric,
                    'vote_metric_value': vote.vote_metric_value,
                    'votes': vote.votes
                } for vote in query.all()
            ]
            return results
        finally:
            session.close()

    def load_database(self, db_path: str = None):
        if db_path is None:
            db_path = self.db_path

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.db_path = db_path

        self.engine = create_engine(f'sqlite:///{db_path}')

        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

        return self

    def directory_to_db(self, dir=None):
        if not dir:
            root_path = FrameworksPath / "ExtenTruth"
        else:
            root_path = dir

        added_graphs = {
            "BarabasiAlbert": 0,
            "ErdosRenyi": 0,
            "total": 0
        }

        for graph_type in ['BarabasiAlbert', 'ErdosRenyi']:
            graph_type_path = root_path / graph_type

            for num_args_dir in graph_type_path.iterdir():
                if not num_args_dir.is_dir():
                    continue

                num_arguments = int(num_args_dir.name.split('_')[-1])

                probability_type = 'cycle_probability' if graph_type == 'BarabasiAlbert' else 'attack_probability'

                for prob_dir in num_args_dir.iterdir():
                    if not prob_dir.is_dir():
                        continue

                    probability_value = float(prob_dir.name.split('_')[-1])

                    for apx_file in prob_dir.glob('AF_*.apx'):
                        graph_name = int(apx_file.name.split('.')[0].split('_')[-1])
                        try:
                            with open(apx_file, 'r') as f:
                                graph_apx = f.read()

                            graph_id = self.insert_graph(
                                graph_type=graph_type,
                                num_arguments=num_arguments,
                                probability_type=probability_type,
                                probability_value=probability_value,
                                graph_name=graph_name,
                                graph_apx=graph_apx
                            )

                            added_graphs[graph_type] += 1
                            added_graphs['total'] += 1

                        except Exception as e:
                            print(f"Error processing {apx_file}: {e}")

        return added_graphs

    def generate_gilbert_graphs(self, graph_type, num_graphs=100,
                                sizes=[5, 10, 15, 20, 30, 40], params=[0.1, 0.25, 0.5, 0.75, 0.9],
                                batch_size=100):
        session = self.Session()

        try:
            afs_to_insert = []
            for size, number, param in tqdm(product(sizes, range(num_graphs), params),
                                  total=len(sizes)*num_graphs*len(params), desc="Total progress"):
                g = OBAF()
                len_ext = 0
                while len_ext < 2:
                    g.generate_random_graph(graph_type, num_args=size, proba_attack=param)
                    graph_data = {
                        'graph_type': graph_type,
                        'num_arguments': size,
                        'graph_generation_metrics': {"attack_probability": param},
                        'graph_name': number + 1,
                        'graph_apx': str(g.af),
                        'extensions': self.json_seriasable_extensions_dict(g.af.get_extensions()),
                        'pref_num_extensions': len(g.af.get_extensions('pref')),
                        'comp_num_extensions': len(g.af.get_extensions('comp')),
                        'pref_ground_truth': self.json_seriasanle_extension(
                            random.choice(g.af.get_extensions('pref'))),
                        'comp_ground_truth': self.json_seriasanle_extension(
                            random.choice(g.af.get_extensions('comp')))
                    }
                    len_ext = len(g.af.get_extensions("pref"))
                afs_to_insert.append(deepcopy(graph_data))
                if len(afs_to_insert) >= batch_size:
                    session.bulk_insert_mappings(GraphModel, afs_to_insert)
                    session.commit()
                    afs_to_insert = []

            if afs_to_insert:
                session.bulk_insert_mappings(GraphModel, afs_to_insert)
                session.commit()

        except SQLAlchemyError as e:
            print(f"Error while generating graphs: {e}")
            session.rollback()
        finally:
            session.close()

    def generate_watts_strogatz_graphs(self, sizes=None, pcs=None, prs=None, batch_size=100):
        session = self.Session()
        graph_generator = AFBenchGraphGenerator()

        try:
            afs_to_insert = []
            if sizes is None:
                sizes = data_structure['WattsStrogatz']['num_args']
            if pcs is None:
                pcs = data_structure['WattsStrogatz']['graph_generation_metrics']['prob_cycles']
            if prs is None:
                prs = data_structure['WattsStrogatz']['graph_generation_metrics']['prob_rewiring']
            ks = data_structure['WattsStrogatz']['graph_generation_metrics']['k_nearest_neighbor']
            ms = data_structure['WattsStrogatz']['graph_generation_metrics']['number_of_afs']
            for size in sizes:
                print("Generating WattsStrogatz graphs for size {}".format(size))
                this_ks = ks(size)
                this_ms = ms(size)
                for pc, pr, k, i in tqdm(product(pcs, prs, this_ks, range(this_ms)),
                                   total=len(pcs)*len(prs)*len(this_ks)*this_ms,
                                   desc=f"{size} args progress"):
                    len_ext = 0
                    j = 0
                    while len_ext < 2 and j < 50:
                        graph_data = None
                        g = OBAF()
                        g = graph_generator.generate_WS_af(num_args=size, prob_cycles=pc,
                                                                 prob_rewiring=pr, k=k)
                        graph_data = {
                            'graph_type': "WattsStrogatz",
                            'num_arguments': size,
                            'graph_generation_metrics': {
                                "prob_cycles": pc,
                                "prob_rewiring": pr,
                                "k_nearest_neighbor": k
                            },
                            'graph_name': i+1,
                            'graph_apx': str(g.af),
                            'extensions': self.json_seriasable_extensions_dict(g.af.get_extensions()),
                            'pref_num_extensions': len(g.af.get_extensions('pref')),
                            'comp_num_extensions': len(g.af.get_extensions('comp')),
                            'pref_ground_truth': self.json_seriasanle_extension(random.choice(g.af.get_extensions('pref'))),
                            'comp_ground_truth': self.json_seriasanle_extension(random.choice(g.af.get_extensions('comp')))
                        }
                        len_ext = len(g.af.get_extensions("pref"))
                        j += 1
                    if len_ext >= 2:
                        afs_to_insert.append(deepcopy(graph_data))
                    if len(afs_to_insert) >= batch_size:
                        session.bulk_insert_mappings(GraphModel, afs_to_insert)
                        session.commit()
                        afs_to_insert = []

            if afs_to_insert:
                session.bulk_insert_mappings(GraphModel, afs_to_insert)
                session.commit()

        except SQLAlchemyError as e:
            print(f"Error while generating graphs: {e}")
            session.rollback()
        finally:
            session.close()

    def json_seriasanle_extension(self, extension):
        return [x for x in extension.arguments]

    def json_desseriasanle_extension(self, extension):
        return Extension(extension)

    def json_seriasable_extensions_dict(self, ext_dict):
        new_dict = {}
        for semantic in ['pref', 'comp', 'stab']:
            new_dict[semantic] = []
            for extension in ext_dict[semantic]:
                new_dict[semantic].append(self.json_seriasanle_extension(extension))
        return new_dict

    def json_desseriasable_extensions_dict(self, ext_dict):
        new_dict = {}
        new_dict['extensions'] = {}
        for semantic in ['pref', 'comp', 'stab']:
            new_dict['extensions'][semantic] = []
            for extension in ext_dict[semantic]:
                new_dict['extensions'][semantic].append(self.json_desseriasanle_extension(extension))
        return new_dict

    def generate_vote(self, graph_type, semantics, batch_size=2000):
        session = self.Session()
        try:
            graphs = session.query(GraphModel).filter(GraphModel.graph_type == graph_type).all()

            for graph in tqdm(graphs):
                if not semantics:
                    semantics = vote_params["semantics"]
                votes_to_insert = []
                for semantic in semantics:
                    for vote_generator_type, vote_details in vote_params["vote_types"].items():
                        metric = vote_details["metric"]
                        values = vote_details["values"]

                        g = OBAF()
                        g.populate(text=graph.graph_apx, isfile=False, read_votes=False)
                        g.af._solution = self.json_desseriasable_extensions_dict(graph.extensions)
                        if semantic == "pref":
                            gt_ext = self.json_desseriasanle_extension(graph.pref_ground_truth)
                        elif semantic == "comp":
                            gt_ext = self.json_desseriasanle_extension(graph.comp_ground_truth)
                        if gt_ext is None:
                            raise ValueError("Ground truth extension is empty")
                        gt = "-".join(gt_ext)

                        if vote_generator_type == _absaf_votes:
                            continue
                        #     for vote_distrib in self.vote_distributions:
                        #         for vote_type in self.vote_types:
                        #             strategy = ABSAFVoteStrategy(_num_votes, semantic)
                        #             g.set_vote_generation_strategy(strategy)
                        #             for value in values:
                        #                 g.generate_votes(vote_distrib, mean=value, gt_ext=gt_ext)
                        #                 vote_data = {
                        #                     'graph_id': graph.id,
                        #                     'semantic': semantic,
                        #                     'num_extensions': len(g.af.get_extensions(semantic)),
                        #                     'extension': gt,
                        #                     'vote_generator_type': vote_generator_type,
                        #                     'vote_type': vote_type,
                        #                     'votes_distribution': vote_distrib,
                        #                     'vote_metric': metric,
                        #                     'metric_value': value,
                        #                     'votes': g.write_votes_to_apx_format()
                        #                 }
                        #
                        #                 votes_to_insert.append(vote_data)
                        #                 if len(votes_to_insert) >= batch_size:
                        #                     session.bulk_insert_mappings(ABSAFVoteModel, votes_to_insert)
                        #                     session.commit()
                        #                     votes_to_insert = []

                        elif vote_generator_type == _our_votes:
                            for vote_distrib in self.vote_distributions:
                                for vote_type in self.vote_types:
                                    strategy = BasicStrategy(_num_votes, "pref")
                                    g.set_vote_generation_strategy(strategy)
                                    for value in values:
                                        if vote_type == "nozeros":
                                            g.generate_votes(type=vote_distrib, mean=value, gt_ext=gt_ext,
                                                             no_abs=True, is_consistent=False)
                                        elif vote_type == "zeros":
                                            g.generate_votes(type=vote_distrib, mean=value, gt_ext=gt_ext,
                                                             no_abs=False, is_consistent=False)
                                        vote_data = {
                                            'graph_id': graph.id,
                                            'semantic': semantic,
                                            'vote_generator_type': vote_generator_type,
                                            'vote_type': vote_type,
                                            'votes_distribution': vote_distrib,
                                            'vote_metric': metric,
                                            'vote_metric_value': value,
                                            'votes': g.write_votes_to_apx_format()
                                        }

                                        votes_to_insert.append(vote_data)
                                        if len(votes_to_insert) >= batch_size:
                                            session.bulk_insert_mappings(VoteModel, votes_to_insert)
                                            session.commit()
                                            votes_to_insert = []

                if votes_to_insert:
                    session.bulk_insert_mappings(VoteModel, votes_to_insert)
                    session.commit()

        except SQLAlchemyError as e:
            print(f"Error while generating votes: {e}")
            session.rollback()
        finally:
            session.close()

    def add_integer_column(self, table_name, column_name, column_value):
        """Add num_extensions column to the existing votes table"""
        from sqlalchemy.sql import text

        session = self.Session()
        try:
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

    def add_string_column(self, table_name, column_name, default_value):
        session = self.Session()
        try:
            session.execute(text(f"""
                ALTER TABLE {table_name} ADD COLUMN {column_name} TEXT DEFAULT {default_value}
            """))
            session.commit()
            print(f"Successfully added string column '{column_name}' to '{table_name}' table.")
        except Exception as e:
            session.rollback()
            print(f"Error adding string column '{column_name}': {e}")
        finally:
            session.close()

    def add_json_column(self, table_name, column_name, default_value=None):
        """Adds a new JSON column to an existing table with an optional default value"""

        session = self.Session()
        try:
            if default_value is not None:
                default_clause = f" DEFAULT '{json.dumps(default_value)}'::jsonb"
            else:
                default_clause = ""

            comm = text(f"""
                ALTER TABLE "{table_name}" 
                ADD COLUMN "{column_name}" JSONB{default_clause}
            """)
            session.execute(comm)
            session.commit()
            print(f"Successfully added {column_name} column to {table_name} as JSONB.")
        except ProgrammingError as pe:
            session.rollback()
            if 'already exists' in str(pe):
                print(f"Column '{column_name}' already exists in table '{table_name}'.")
            else:
                print(f"SQL error: {pe}")
        except Exception as e:
            session.rollback()
            print(f"Error adding JSON column: {e}")
        finally:
            session.close()

    def delete_all_rows(self, table_model):
        """Delete all rows from a specified table"""
        session = self.Session()
        try:
            session.query(table_model).delete()
            session.commit()
            print(f"All rows deleted from {table_model.__tablename__}")
        except Exception as e:
            session.rollback()
            print(f"Error deleting rows: {e}")
        finally:
            session.close()

    def plot_extensions_by_arguments_from_db(self, extension_type: str = "pref", aggregation: str = "mean",
                                             graph_type=None, log_threshold: int = 1000):
        if extension_type not in ("pref", "comp"):
            raise ValueError("extension_type must be 'pref' or 'comp'")

        column_name = f"{extension_type}_num_extensions"

        session = self.Session()

        if graph_type is not None:
            result = (session.query(GraphModel.num_arguments, getattr(GraphModel, column_name))
                      .filter(GraphModel.graph_type == graph_type)
                      .all())
        else:
            result = (session.query(GraphModel.num_arguments, getattr(GraphModel, column_name))
                      .all())

        df = pd.DataFrame(result, columns=["arguments", "extensions"])
        df = df.dropna(subset=["extensions"])

        group_counts = df.groupby("arguments").size().reset_index(name="count")

        if aggregation == "mean":
            agg_df = df.groupby("arguments")["extensions"].mean()
        elif aggregation == "min":
            agg_df = df.groupby("arguments")["extensions"].min()
        elif aggregation == "max":
            agg_df = df.groupby("arguments")["extensions"].max()
        else:
            raise ValueError("aggregation must be one of: 'mean', 'min', 'max'")

        agg_df = agg_df.reset_index()

        agg_df = pd.merge(agg_df, group_counts, on="arguments")

        norm = plt.Normalize(agg_df["count"].min() - 10, agg_df["count"].max() + 10)
        cmap = plt.cm.viridis
        colors = [cmap(norm(count)) for count in agg_df["count"]]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(agg_df["arguments"], agg_df["extensions"], color=colors, width=3)

        for bar, value in zip(bars, agg_df["extensions"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.01), f'{value:.1f}', ha='center',
                    va='bottom', fontsize=9)

        if agg_df["extensions"].max() > log_threshold:
            ax.set_yscale('log')
            y_label = f"Number of Extensions (log scale)"
        else:
            y_label = "Number of Extensions"

        ax.set_title(f"Number of Extensions by Number of Arguments ({aggregation})")
        ax.set_xlabel("Number of Arguments")
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Number of Graphs in Group")

        plt.tight_layout()
        plt.show()

    def combine_columns_to_json(self):
        session = self.Session()
        try:
            rows = session.query(GraphModel).all()

            for row in tqdm(rows):
                key = row.probability_type
                value = row.probability_value

                row.graph_generation_metrics = {key: value}

            session.commit()
            print("Successfully combined columns into JSON.")
        except Exception as e:
            session.rollback()
            print(f"An error occurred: {e}")
        finally:
            session.close()

    def fill_extension_columns(self, batch=100):
        session = self.Session()
        try:
            rows = session.query(GraphModel).yield_per(batch)

            batch_rows = []
            for i, row in enumerate(tqdm(rows), 1):
                g = OBAF()
                g.populate(text=row.graph_apx, isfile=False, read_votes=False)

                row.extensions = self.json_seriasable_extensions_dict(g.af.get_extensions())
                flag_modified(row, "extensions")

                row.pref_num_extensions = len(g.af.get_extensions('pref'))
                row.comp_num_extensions = len(g.af.get_extensions('comp'))

                row.pref_ground_truth = None
                flag_modified(row, "pref_ground_truth")

                row.comp_ground_truth = None
                flag_modified(row, "comp_ground_truth")

                batch_rows.append(deepcopy(row))

                if i % batch == 0:
                    session.commit()
                    batch_rows.clear()

            if batch_rows:
                session.commit()
                batch_rows.clear()

            print("Successfully updated extension columns in batches.")
        except Exception as e:
            session.rollback()
            print(f"An error occurred: {e}")
        finally:
            session.close()

    def fill_ground_truth_from_votes(self, batch=100):
        session = self.Session()
        try:
            graphs = session.query(GraphModel).yield_per(batch)

            for i, graph in enumerate(tqdm(graphs), 1):
                pref_vote = (
                    session.query(VoteModel)
                    .filter_by(graph_id=graph.id, semantic="pref")
                    .first()
                )
                comp_vote = (
                    session.query(VoteModel)
                    .filter_by(graph_id=graph.id, semantic="comp")
                    .first()
                )

                graph.pref_ground_truth = pref_vote.ground_truth.split("-")
                flag_modified(graph, "pref_ground_truth")
                graph.comp_ground_truth = comp_vote.ground_truth.split("-")
                flag_modified(graph, "comp_ground_truth")

                if i % batch == 0:
                    session.commit()

            session.commit()
            print("Ground truth successfully filled from votes.")
        except Exception as e:
            session.rollback()
            print(f"An error occurred: {e}")
        finally:
            session.close()

    def drop_column(self, table_name, column_name):
        """
        Drops a column from a table in the database.

        :param table_name: str, name of the table (e.g., 'graphs')
        :param column_name: str, name of the column to drop (e.g., 'old_column')
        """
        from sqlalchemy.sql import text

        session = self.Session()
        try:
            session.execute(text(f'ALTER TABLE {table_name} DROP COLUMN {column_name}'))
            session.commit()
            print(f"Successfully dropped column '{column_name}' from '{table_name}' table.")
        except Exception as e:
            session.rollback()
            print(f"Error dropping column '{column_name}' from '{table_name}': {e}")
        finally:
            session.close()

    def copy_graphs_to_new_db(self, new_db_path, graph_type_filter, num_graphs_per_combo):
        """
        Copy graphs from the current DB to a new DB file based on parameter combinations.

        :param new_db_path: Path to new SQLite DB file (e.g. 'subset.db')
        :param graph_type_filter: Graph type string to filter on (e.g. 'barabasi albert')
        :param num_graphs_per_combo: How many graphs per parameter combo to copy
        """

        new_engine = create_engine(f"sqlite:///{new_db_path}")
        Base.metadata.create_all(new_engine)
        NewSession = sessionmaker(bind=new_engine)
        new_session = NewSession()

        old_session = self.Session()

        try:
            if graph_type_filter == "BarabasiAlbert":
                graph_sizes = data_structure[graph_type_filter]["num_args"]
                probabilities = data_structure[graph_type_filter]["graph_generation_metrics"]["prob_cycles"]
                proba_name = "cycle_probability"
            if graph_type_filter == "Gilbert":
                graph_sizes = data_structure[graph_type_filter]["num_args"]
                probabilities = data_structure[graph_type_filter]["graph_generation_metrics"]["prob_attacks"]
                proba_name = "attack_probability"
            if graph_type_filter == "WattsStrogatz":
                # graph_sizes = data_structure[graph_type_filter]["num_args"]
                graph_sizes = [5, 10]
                probabilities = []
                for size in graph_sizes:
                    k_function_res = data_structure[graph_type_filter]["graph_generation_metrics"]["k_nearest_neighbor"](size)
                    pcs = data_structure[graph_type_filter]["graph_generation_metrics"]["prob_cycles"]
                    prs = data_structure[graph_type_filter]["graph_generation_metrics"]["prob_rewiring"]
                    m = math.ceil(num_graphs_per_combo/(
                        len(pcs)*len(prs)*len(k_function_res)
                    ))
                    probabilities.extend(product([size],
                                                 pcs, prs,
                                                 k_function_res, [m]))
                proba_name = ("prob_cycles", "prob_rewiring", "k_nearest_neighbor")

            if graph_type_filter == "WattsStrogatz":
                prod = probabilities
                tot = len(prod)
            else:
                prod = product(graph_sizes, probabilities)
                tot = len(graph_sizes)*len(probabilities)
            for item in tqdm(prod,
                                    total=tot,
                                    desc="Total progress"):
                size = item[0]
                if graph_type_filter == "WattsStrogatz":
                    proba = item[1:]
                    graphs = (
                        old_session.query(GraphModel)
                        .filter_by(graph_type=graph_type_filter,
                                   num_arguments=size,
                                   graph_generation_metrics={proba_name[0]:proba[0],
                                                             proba_name[1]:proba[1],
                                                             proba_name[2]:proba[2]})
                        .all()
                    )
                    num_graphs = proba[3]
                else:
                    proba = item[1]
                    graphs = (
                        old_session.query(GraphModel)
                        .filter_by(graph_type=graph_type_filter,
                                   num_arguments=size,
                                   graph_generation_metrics={proba_name:proba})
                        .all()
                    )
                    num_graphs = num_graphs_per_combo

                selected = random.sample(graphs, min(num_graphs, len(graphs)))

                for graph in selected:
                    # Clone graph
                    new_graph = GraphModel(
                        id=graph.id,
                        graph_type=graph.graph_type,
                        num_arguments=graph.num_arguments,
                        graph_generation_metrics=graph.graph_generation_metrics,
                        graph_name=graph.graph_name,
                        graph_apx=graph.graph_apx,
                        extensions=graph.extensions,
                        pref_num_extensions=graph.pref_num_extensions,
                        comp_num_extensions=graph.comp_num_extensions,
                        pref_ground_truth=graph.pref_ground_truth,
                        comp_ground_truth=graph.comp_ground_truth,
                    )
                    new_session.add(new_graph)

                    for vote in graph.votes:
                        new_vote = VoteModel(
                            graph_id=new_graph.id,
                            semantic=vote.semantic,
                            vote_generator_type=vote.vote_generator_type,
                            vote_type=vote.vote_type,
                            votes_distribution=vote.votes_distribution,
                            vote_metric=vote.vote_metric,
                            vote_metric_value=vote.vote_metric_value,
                            votes=vote.votes
                        )
                        new_session.add(new_vote)

                new_session.commit()

            print("Graphs copied successfully to new database.")
        except Exception as e:
            new_session.rollback()
            print(f"Error copying graphs: {e}")
        finally:
            old_session.close()
            new_session.close()

    def shift_graph_ids(self, shift_by):
        """
        Shift all graph IDs by a given value and update foreign key references in votes.
        """
        session = self.Session()
        try:
            print("Updating votes.graph_id...")
            session.execute(text(f"""
                UPDATE votes
                SET graph_id = graph_id + :shift_by
            """), {"shift_by": shift_by})

            print("Updating graphs.id...")
            session.execute(text(f"""
                UPDATE graphs
                SET id = id + :shift_by
            """), {"shift_by": shift_by})

            # Optional: if you also have absaf_votes
            print("Updating absaf_votes.graph_id (if applicable)...")
            session.execute(text(f"""
                UPDATE absaf_votes
                SET graph_id = graph_id + :shift_by
            """), {"shift_by": shift_by})

            session.commit()
            print(f"✅ Successfully shifted all graph IDs by {shift_by}")
        except Exception as e:
            session.rollback()
            print(f"❌ Error while shifting IDs: {e}")
        finally:
            session.close()

    def delete_graphs_by_type(self, graph_type_to_delete):
        session = self.Session()
        try:
            print(f"Deleting graphs with graph_type = '{graph_type_to_delete}'")

            graph_ids = session.query(GraphModel.id).filter(
                GraphModel.graph_type == graph_type_to_delete
            ).all()
            graph_ids = [g[0] for g in graph_ids]

            if not graph_ids:
                print("No matching graphs found.")
                return

            print("Deleting related votes...")
            session.query(VoteModel).filter(VoteModel.graph_id.in_(graph_ids)).delete(synchronize_session=False)

            print("Deleting related absaf_votes...")
            session.query(ABSAFVoteModel).filter(ABSAFVoteModel.graph_id.in_(graph_ids)).delete(
                synchronize_session=False)

            print("Deleting graphs...")
            session.query(GraphModel).filter(GraphModel.id.in_(graph_ids)).delete(synchronize_session=False)

            session.commit()
            print(f"Successfully deleted graphs of type '{graph_type_to_delete}'")
        except Exception as e:
            session.rollback()
            print(f"Error while deleting graphs: {e}")
        finally:
            session.close()
