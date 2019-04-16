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


from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, Tuple, TypeVar

import combo
from combo import variable as Variable
import numpy as np

from .action import Action
from .types import GetTest, GetX


if TYPE_CHECKING:
    from typing import Protocol


    class Model(Protocol):

        inf: 'Inf'


    class Inf(Protocol):

        exact: 'Exact'


    class Exact(Protocol):
        ...


class Chooser(ABC):

    @abstractmethod
    def choose_random_actions(self, N: int) -> Action:
        ...

    @abstractmethod
    def choose_bayes_action(self, get_score: GetTest) -> Action:
        ...
