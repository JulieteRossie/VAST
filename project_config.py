# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

ProjectPath = Path(__file__).parent.absolute()
FrameworksPath = ProjectPath / "frameworks"
DatasetPath = FrameworksPath / "AFBenchGen2"
ResultsPath = ProjectPath / "results"