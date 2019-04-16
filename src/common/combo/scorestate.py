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


from typing import Callable

import combo
import numpy as np


def init_test(predictor: combo.base_predictor, test_X: np.ndarray) -> combo.variable:
    return combo.variable(X=test_X, Z=predictor.get_basis(test_X))


class ScoreState:

    def __init__(self, predictor: combo.base_predictor, centering: Callable[[np.ndarray], np.ndarray], mode: Callable[[combo.variable], np.ndarray]) -> None:
        self.predictor = predictor
        self.centering = centering
        self.mode = mode

    def init_test(self, X: np.ndarray) -> combo.variable:
        return init_test(self.predictor, self.centering(X))

    def __call__(self, X: np.ndarray) -> combo.variable:
        test = self.init_test(X)
        test.t = self.mode(test)
        return test
