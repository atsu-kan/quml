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
from typing import Callable, Generator, Iterator, List, Tuple

import numpy as np
import pandas as pd

from src.common.genertools import call
from src.common.math import allclose
from src.pylabzmqinterface.adaptee import Adaptee


class Session(Adaptee):

    def __init__(self, on_session: Iterator[Generator[pd.Series, pd.Series, None]], param_header: np.ndarray, result_header: np.ndarray, initial_param: np.ndarray) -> None:
        super().__init__()
        self._param_header = pd.Index(param_header)
        self._result_header = pd.Index(result_header)
        self._initial_param = pd.Series(initial_param, index=self._param_header)
        self._exps: List[Tuple[pd.Series, Callable[[pd.Series], None]]] = []
        self._on_session = on_session

    def write(self, last_param: np.ndarray, last_result: np.ndarray) -> None:
        for i, (param, write_result) in enumerate(self._exps):
            if allclose(param.values, last_param):
                self._exps.pop(i)
                write_result(pd.Series(last_result, index=self._result_header))
                break

    def create_reader(self) -> Iterator[np.ndarray]:
        last_param = self._initial_param
        for on_experiment in self._on_session:
            param, write_result = call(on_experiment)
            last_param = copy(last_param)
            for i in param.index:
                if i in last_param.index:
                    last_param[i] = param[i]
            self._exps.append((last_param, write_result))
            yield last_param.values
