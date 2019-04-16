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

import combo
import numpy as np


class ScoreMode(ABC):

    @abstractmethod
    def __init__(self, predictor: combo.base_predictor, training: combo.variable) -> None:
        ...

    @abstractmethod
    def __call__(self, test: combo.variable) -> np.ndarray:
        ...


class EI(ScoreMode):

    def __init__(self, predictor: combo.base_predictor, training: combo.variable) -> None:
        self.predictor = predictor
        self.training = training

    def __call__(self, test: combo.variable) -> np.ndarray:
        return combo.search.score.EI(self.predictor, self.training, test)


class PI(ScoreMode):

    def __init__(self, predictor: combo.base_predictor, training: combo.variable) -> None:
        self.predictor = predictor
        self.training = training
        
    def __call__(self, test: combo.variable) -> np.ndarray:
        return combo.search.score.PI(self.predictor, self.training, test)


class TS(ScoreMode):

    def __init__(self, predictor: combo.base_predictor, training: combo.variable) -> None:
        assert(isinstance(predictor, combo.blm.predictor))
        self.w_hat = predictor.blm.sampling(alpha=predictor.config.learning.alpha)
        self.bias = predictor.blm.lik.linear.bias

    def __call__(self, test: combo.variable) -> np.ndarray:
        Psi = test.Z
        return Psi.dot(self.w_hat) + self.bias
