# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import numpy as np
import warnings

from project_config import *
from structure.argumentation_framework import AF
from structure.vote import Vote
from structure.extension import Extension

warnings.filterwarnings("ignore")


class OBAF:
    """
    Opinion Based Argumentation Framework (OBAF).
    Combines a standard Argumentation Framework (AF) with a set of user votes (opinions) as vectors
    """
    af: AF = None
    votes: Optional[List[Vote]] = None
    vote_counter: int = 0
    vote_generator = None

    def __init__(self, vote_generator=None):
        self.af = AF()
        self.vote_generator = vote_generator

    def __str__(self):
        """Returns the string representation of the AF and all votes"""
        strg = str(self.af)
        for vote in self.votes:
            strg += str(vote) + "\n"
        return strg

    def populate(self, filename=None, path=FrameworksPath, read_votes=True, num_votes=None,
                 wrong_arg_name_format=False, text=None, isfile=True):
        """
        Populates the AF and optionally parses votes from a source, aither an APX file (then fill file name and path)
        or a string (then fill in text)

        :param filename: APX file where the AF and votes are stored
        :param path: path to the file
        :param read_votes: boolean, when true the method will fill the OBAF with the votes from the APX file
        :param num_votes: number of votes to be read from the file
        :param wrong_arg_name_format: if ASP causes an error, put this to True to automatically correct arguments' names
        :param text: if the APX is in the format of string put it here
        :param isfile: if reading the APX from a file, and filename and path are filled, this should be Ture,
        otherwise it should be false
        """
        self.votes = None
        self.af.populate(filename, path, wrong_arg_name_format=wrong_arg_name_format, text=text, isfile=isfile)
        if read_votes:
            self.repopulate_vote_files(filename=filename, path=path, num_votes=num_votes, wrong_arg_name_format=wrong_arg_name_format)

    def repopulate_vote_files(self, text=None, filename=None, path=None, num_votes=None, wrong_arg_name_format=False):
        """
        Parses votes from file or text string into the OBAF (contrary to populate, this only fills the votes)

        :param text: if the APX is in the format of string put it here
        :param filename: APX file where the AF and votes are stored
        :param path: path to the file
        :param num_votes: number of votes to be read from the file
        :param wrong_arg_name_format: if ASP causes an error, put this to True to automatically correct arguments' names
        """
        self.vote_counter = 0
        self.votes = None
        if filename is not None and path is not None:
            file = open(path / filename, 'r')
            lines = file.readlines()
        elif text is not None:
            lines = text.split("\n")
        else:
            raise ValueError("No apx was provided")
        j = 0
        for line in lines:
            if 'vote(' in line:
                to_add = Vote('v' + str(self.vote_counter))
                tmp = line[5:-3]
                tmp = tmp.replace(')', '')
                tmp = tmp.replace('(', '')
                tab = tmp.split(',')
                tab = tab[1:]
                for i, a in self.af.arguments.items():
                    if wrong_arg_name_format:
                        arg_name = self.af.new_to_old_mapping.get(a.name)
                    else:
                        arg_name = a.name
                    if arg_name in tab:
                        i = tab.index(arg_name)
                        to_add.add(a.name, int(tab[i + 1]))
                    else:
                        to_add.add(a.name, 0)
                self.add_vote(to_add)
                j += 1
                if num_votes is not None and j >= num_votes:
                    break

    def remove_votes(self):
        """Clears all existing votes"""
        self.votes = None
        self.vote_counter = 0

    def add_vote(self, vote):
        """Adds a single vote to the OBAF"""
        if self.votes is None:
            self.votes = list()
        self.votes.append(vote)
        self.vote_counter += 1

    def add_vote_from_ext(self, ext: Extension):
        """Converts an extension into a vote (+1 for in, -1 for out) and adds it to the OBAF"""
        to_add = Vote('v' + str(self.vote_counter))
        for arg in self.af.arguments:
            if arg in ext:
                to_add.add(arg, 1)
            else:
                to_add.add(arg, -1)
        self.add_vote(to_add)

    def draw_votes(self, fig, ax):
        """Draws the matrix of votes"""
        args = list(self.af.arguments.keys())
        data = [x.vote_dict for x in self.votes]
        data = [list(x.values()) for x in data]
        data = np.array(data)
        self.af.graph.draw_votes(fig, ax, data, args)

    def draw(self, votes=True, values=None, undirected_edges=[]):
        """Visualizes the AF graph and optionally the votes"""
        args = list(self.af.arguments.keys())
        args.sort()  # This ensures that argument order is always the same

        data = None

        if votes and self.votes:
            # Extract votes in the same order as 'args'
            data = []
            for vote in self.votes:
                # Make sure to extract values in the order of 'args'
                ordered_vote_values = [vote.vote_dict[arg] for arg in args]
                data.append(ordered_vote_values)

        self.af.graph.draw(values, undirected_edges, data, args)

    def set_vote_generation_strategy(self, strategy):
        """Set a different vote generation strategy dynamically"""
        self.vote_generator = strategy

    def generate_votes(self, type: str, mean, gt_ext, std=None, no_abs=True, is_consistent=False):
        """Generate votes using the current strategy."""
        if type == "normal":
            return self.vote_generator.generate_normal_votes(self, mean, std, gt_ext, None)
        if type == "mean":
            return self.vote_generator.generate_mean_votes(self, mean, gt_ext, None, no_abs, is_consistent)
        elif type == "uniform":
            return self.vote_generator.generate_uniform_votes(self, mean, gt_ext, None)

    def try_all(self, reliabilities, gt=None, generation_type="uniform"):
        """Test the current strategy across different reliability values."""
        return self.vote_generator.try_all(self, reliabilities, gt, generation_type=generation_type)

    def write_votes_to_file(self, path, filename='votes.apx'):
        """Exports the current votes to an APX file"""
        path = Path(path)
        with open(path / filename, 'w') as file:
            for idx, vote in enumerate(self.votes):
                # Construct the vote string
                vote_str = f"vote(a{idx + 1}"
                for arg, val in vote.vote_dict.items():
                    vote_str += f",({arg},{val})"
                vote_str += ").\n"

                # Write the vote string to the file
                file.write(vote_str)

    def write_votes_to_apx_format(self):
        """Formats the current votes into APX string format"""
        result = ""
        for idx, vote in enumerate(self.votes):
            # Construct the vote string
            vote_str = f"vote(a{idx + 1}"
            for arg, val in vote.vote_dict.items():
                vote_str += f",({arg},{val})"
            vote_str += ").\n"
            result += vote_str
        return result

    def generate_random_graph(self, generation_type, min_ext=2, **kwargs):
        """Generates a random AF that has at least `min_ext` preferred extensions"""
        len_ext = 0
        while len_ext < min_ext:
            self.af = AF()
            self.af.random_af(generation_type, **kwargs)
            len_ext = len(self.af.get_extensions("pref"))
