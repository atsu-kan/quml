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


from typing import Optional

import combo
import numpy as np

from .predictor import Predictor


class Policy:

    def __init__(self) -> None:
        self.training = combo.variable()
        self.history = combo.search.discrete.history()

    @property
    def training_len(self) -> int:
        if self.training.t is not None:
            return self.training.t.shape[0]
        else:
            return 0

    def write(self, X: np.ndarray, t: np.ndarray, is_disp: bool = True) -> None:
        st = self.history.total_num_search + 1
        self.history.write(t, np.arange(st, st + t.shape[0]))
        self.training.add(X=X, t=t)
        if is_disp:
            combo.search.utility.show_search_results(self.history, t.shape[0])

    def learn(self, num_rand_basis: int, config: Optional[combo.misc.set_config] = None) -> Predictor:
        return Predictor(self.training, num_rand_basis, config)
