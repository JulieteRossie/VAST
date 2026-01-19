# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import ArrowStyle
import numpy as np
from typing import List


class Graph:
    """
    Wrapper around NetworkX for AF graph manipulation and visualization.
    If networks needs to be replaced this should be done in this wrapper
    """
    graph = None

    def __init__(self, edges, nodes):
        self.graph = nx.DiGraph()
        assert (edges is not None and nodes is not None)
        if edges:
            self.add_edges(edges)
        if nodes:
            self.add_nodes(nodes)

    def add_edges(self, edges):
        """Adds edges to the graph"""
        self.graph.add_edges_from(edges)

    def add_nodes(self, nodes):
        """Adds nodes to the graph"""
        self.graph.add_nodes_from(nodes)

    def create_graph(self, val_map=None):
        """Map node values for coloring or processing"""
        if val_map is None:
            values = [1.0 for _ in self.graph.nodes()]
        else:
            values = [val_map.get(node, 0.25) for node in self.graph.nodes()]
        return values

    def draw_graph(self, ax, values=None, undirected_edges=[]):
        """Draws the argumentation graph on the specified axes"""
        pos = nx.spring_layout(self.graph, seed=38, k=1.5)  # positions for all nodes - seed for reproducibility

        if values is not None:
            nx.draw_networkx_nodes(self.graph, pos, cmap=plt.get_cmap('Wistia'),
                                   node_color=values, node_size=800, ax=ax)
        else:
            nx.draw_networkx_nodes(self.graph, pos, node_color='#FFFFFF', node_size=1000, edgecolors='#000000', ax=ax)

        nx.draw_networkx_labels(self.graph, pos, ax=ax)

        directed_edges = [edge for edge in self.graph.edges() if edge not in undirected_edges]
        arrow_style = ArrowStyle("->", head_length=0.5, head_width=0.25)
        nx.draw_networkx_edges(self.graph, pos, edgelist=directed_edges, width=2.5, arrows=True,
                               arrowsize=25, arrowstyle=arrow_style, alpha=0.6, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=undirected_edges, width=2.5, arrows=False, alpha=0.6, ax=ax)

        plt.tight_layout(pad=1)

    def draw_votes(self, fig, ax, data, args):
        """Visualizes voting data as a color-coded matrix"""
        cmap = colors.ListedColormap(['#F9766E', 'white', '#00BFC4'])
        bounds = [-1, -0.000001, 0.0000001, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        cax = ax.matshow(data, interpolation='nearest', cmap=cmap, norm=norm)
        fig.colorbar(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1, -0.000001, 0.0000001, 1])

        ax.set_xticks(range(len(args)))
        ax.set_xticklabels(args)

        for (i, j), z in np.ndenumerate(data):
            # if bug try this if data[i][j] > 0.5:
            if z > 0.5:
                ax.text(j, i, 'yes', ha='center', va='center')
            elif z < -0.5:
                ax.text(j, i, 'no', ha='center', va='center')

    def draw(self, values=None, undirected_edges=List, votes_data=None, args=None):
        """Plots the graph, optionally alongside vote visualizations"""
        # plt.clf()

        if votes_data is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        else:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        self.draw_graph(ax, values, undirected_edges)

        if votes_data is not None and args is not None:
            self.draw_votes(fig, ax2, votes_data, args)

        plt.show()

    def generate_random(self, generation_type, **kwargs):
        """Generates random graphs using Gilbert or Stochastic Block models"""
        if generation_type == "Gilbert":
            num_args = kwargs.get("num_args", None)
            proba_attack = kwargs.get("proba_attack", None)
            try:
                self.graph = nx.gnp_random_graph(n=num_args, p=proba_attack, directed=True, seed=np.random)
            except TypeError:
                print("ERROR: Graph - generate random: No values given from num_args and proba_attack")

        elif generation_type == "stochastic_block":
            blocks = kwargs.get("blocks", None)
            densities = kwargs.get("densities", None)
            try:
                self.graph = nx.stochastic_block_model(sizes=blocks, p=densities,
                                                       directed=True, seed=np.random)
            except TypeError:
                print("ERROR: Graph - generate random: No values given from blocks and densities")

        else:
            print("ERROR: Graph - generate random: no such graph type")