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


from typing import Callable, Tuple

import numpy as np
import pandas as pd


class Experiment:

    def __init__(self, param_header: pd.Index, result_header: pd.Index, simulate: Callable[[pd.Series], pd.Series], delay_size: int) -> None:
        super().__init__()
        self._param_header = param_header
        self._result_header = result_header
        self._simulate = simulate
        self._exp = [(pd.Series(np.nan, index=self._param_header), pd.Series(np.nan, index=self._result_header)) for _ in range(delay_size)]

    def __call__(self, param: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _param = pd.Series(param, index=self._param_header)
        self._exp.append((_param, self._simulate(_param)))
        _param, _result = self._exp.pop(0)
        return _param.values, _result.values
