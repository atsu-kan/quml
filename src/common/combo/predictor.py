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


from typing import Callable, Optional

import combo
import numpy as np

from .centering import Centering
from .scoremode import EI, PI, TS
from .scorestate import ScoreState


def init_config(config: Optional[combo.misc.set_config]) -> combo.misc.set_config:
    if config is not None:
        return config
    else:
        return combo.misc.set_config()


def init_predictor(num_rand_basis: int, config: Optional[combo.misc.set_config]) -> combo.base_predictor:
    is_rand_expans = False if num_rand_basis == 0 else True
    if is_rand_expans:
        return combo.blm.predictor(init_config(config))
    else:
        return combo.gp.predictor(init_config(config))


def learn(predictor: combo.base_predictor, training: combo.variable, num_rand_basis: int) -> None:
    print_params = predictor.model.prior.cov.print_params
    predictor.model.prior.cov.print_params = lambda: None
    try:
        predictor.fit(training, num_rand_basis)
        print_params()
    finally:
        del predictor.model.prior.cov.print_params
    training.Z = predictor.get_basis(training.X)
    predictor.prepare(training)


def update(predictor: combo.base_predictor, new_data: combo.variable) -> None:
    predictor.update(None, new_data)


class Predictor:

    def __init__(self, training: combo.variable, num_rand_basis: int, config: Optional[combo.misc.set_config] = None) -> None:
        self.config = init_config(config)
        self.new_data = combo.variable()
        self.centering = Centering(training.X, self.config.learning.epsilon)
        self.training = combo.variable(X=self.centering(training.X), t=training.t)
        self._predictor = init_predictor(num_rand_basis=num_rand_basis, config=config)
        learn(self.predictor, self.training, num_rand_basis=num_rand_basis)

    @property
    def predictor(self) -> combo.base_predictor:
        if self.new_data.t is not None and self.new_data.t.shape[0] > 0:
            update(self._predictor, self.new_data)
            self.training.add(X=self.new_data.X, t=self.new_data.t, Z=self.new_data.Z)
            self.new_data = combo.variable()
        return self._predictor

    def write(self, test:combo.variable, t: np.ndarray) -> None:
        self.new_data.add(X=test.X, t=t, Z=test.Z)

    def get_score(self, score: str) -> Callable[[np.ndarray], combo.variable]:
        if score == 'EI':
            return self._init_score(EI)
        elif score == 'PI':
            return self._init_score(PI)
        elif score == 'TS':
            return self._init_score(TS)
        else:
            raise NotImplementedError('mode must be EI, PI or TS.')

    def _init_score(self, mode: Callable[[combo.base_predictor, combo.variable], Callable[[combo.variable], np.ndarray]]) -> Callable[[np.ndarray], combo.variable]:
        return ScoreState(self.predictor, self.centering, mode(self.predictor, self.training))
