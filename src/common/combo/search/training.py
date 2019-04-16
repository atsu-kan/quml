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


import combo
import numpy as np


class Training:

    def __init__(self) -> None:
        self._training = combo.variable()
        self._new_data = combo.variable()
        self._history = combo.search.discrete.history()

    def write(self, test: combo.variable, t: np.ndarray) -> None:
        self._new_data.add(X=test.X, t=t, Z=test.Z)
        self._history.write(t, np.arange(t.shape[0]) + self._history.total_num_search)
