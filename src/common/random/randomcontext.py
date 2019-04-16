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


from contextlib import ContextDecorator
from functools import partial
import random
from typing import Any, Callable, List

import numpy as np


class RandomContext(ContextDecorator):

    _contexts: List['RandomContext']

    def __init__(self, seed: Any = None) -> None:
        super().__init__()
        self._set_to_global: Callable[[], None] = partial(self._set_seed_to_global, seed)

    def _set_from_global(self) -> None:
        self._set_to_global = partial(self._set_state_to_global, random.getstate(), np.random.get_state())

    @staticmethod
    def _set_seed_to_global(seed: Any) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _set_state_to_global(py_state: Any, np_state: Any) -> None:
        random.setstate(py_state)
        np.random.set_state(np_state)

    def __enter__(self) -> Any:
        self._contexts[-1]._set_from_global()
        self._set_to_global()
        self._contexts.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._set_from_global()
        self._contexts.pop()._set_to_global()
        return False

        
RandomContext._contexts = [RandomContext()]
