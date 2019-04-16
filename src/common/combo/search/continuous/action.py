# Copyright 2019 AIST
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from copy import copy
from typing import Any, Optional

from combo import variable as Variable
import numpy as np


class Action:

    def __init__(self, X: np.ndarray, test: Variable) -> None:
        self.X = X
        self.test = test

    def get_subset(self, index: Any) -> 'Action':
        return Action(self.X[index, :], self.test.get_subset(index))

    def add(self, other: 'Action') -> 'Action':
        self.X = np.vstack((self.X, other.X))
        self.test = copy(self.test)
        self.test.add(other.test)
        return self
#
#
# class Action(NamedTuple):
#
#     X: np.ndarray
#     test: Variable
#
#     def get_subset(self, index: Any) -> 'Action':
#         return Action(self.X[index, :], self.test.get_subset(index))
#
#     @staticmethod
#     def concat(actions: 'Iterable[Action]') -> 'Action':
#         return Action(
#             X=np.vstack([action.X for action in actions]),
#             test=Variable(
#                 X=np.vstack([action.test.X for action in actions]),
#                 t=np.hstack([action.test.t for action in actions]),
#                 Z=np.vstack([action.test.Z for action in actions])
#             )
#         )
