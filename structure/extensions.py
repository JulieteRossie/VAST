# Copyright (c) 2025 Juliete Rossie @ CRIL - CNRS
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List

from structure.extension import Extension


class Extensions:
    """Represents a collection of Extension objects"""
    def __init__(self, extensions: Union[List[str], List[Extension]]):
        """Initializes the collection from a list of strings or Extension objects."""
        self.extensions = []
        if isinstance(extensions[0], str):
            for e in extensions:
                self.extensions.append(Extension(e))
        else:
            self.extensions = extensions
