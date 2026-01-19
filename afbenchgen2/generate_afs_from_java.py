# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import os
from tqdm import tqdm

from solver.asp import ASP
from structure.opinion_based_af import OBAF
from experiments.config import data_structure
from project_config import ProjectPath


class AFBenchGraphGenerator:
    def __init__(self):
        self.min_extension = 2
        self.classpath = ("afbenchgen2/bin/main/java" + os.pathsep +
                          "afbenchgen2/lib/commons-cli-1.2.jar" + os.pathsep +
                          "afbenchgen2/lib/gs-core-1.2.jar" + os.pathsep +
                          "afbenchgen2/lib/gs-algo-1.2.jar")

    def generate_BA_af(self, num_args, prob_cycles):
        len_ext = 0
        while len_ext < self.min_extension:
            result = subprocess.run([
                "java", "-cp", self.classpath, "net.sf.jAFBenchGen.jAFBenchGen.Generator",
                "--type", "BarabasiAlbert",
                "--numargs", str(num_args-1),
                "--BA_WS_probCycles", str(prob_cycles)
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: AFBenchGraphGenerator generate_BA_af: unable to generate AF: {result.stderr}")
            else:
                af_content = result.stdout
                g = OBAF()
                g.populate(text=af_content, read_votes=False)
                len_ext = len(g.af.get_extensions("pref"))
        return g

    def generate_WS_af(self, num_args, prob_cycles, prob_rewiring, k):
        len_ext = 0
        i = 0
        while len_ext < self.min_extension and i < 50:
            result = subprocess.run([
                "java", "-cp", self.classpath, "net.sf.jAFBenchGen.jAFBenchGen.Generator",
                "--type", "WattsStrogatz",
                "--numargs", str(num_args),
                "--BA_WS_probCycles", str(prob_cycles),
                "--WS_baseDegree", str(k),
                "--WS_beta", str(prob_rewiring)
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: AFBenchGraphGenerator generate_WS_af: unable to generate AF: {result.stderr}")
            else:
                af_content = result.stdout
                g = OBAF()
                g.populate(text=af_content, read_votes=False)
                len_ext = len(g.af.get_extensions("pref"))
                i += 1
        return g
