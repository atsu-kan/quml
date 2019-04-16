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
from typing import Optional, Tuple

import combo
import numpy as np


def write(new_data: combo.variable, test: combo.variable, t: np.ndarray) -> None:
    new_data.add(X=test.X, t=t, Z=test.Z)


def init_config(config: Optional[combo.misc.set_config] = None) -> combo.misc.set_config:
    if config is not None:
        return config
    else:
        return combo.misc.set_config()


def init_predictor(num_rand_basis: int, config: combo.misc.set_config) -> combo.base_predictor:
    is_rand_expans = False if num_rand_basis == 0 else True
    if is_rand_expans:
        return combo.blm.predictor(config)
    else:
        return combo.gp.predictor(config)


def learn(predictor: combo.base_predictor, training: combo.variable, num_rand_basis: int) -> None:
    training = copy(training)
    predictor.fit(training, num_rand_basis)
    training.Z = predictor.get_basis(training.X)
    predictor.prepare(training)


def update(predictor: combo.base_predictor, new_data: combo.variable) -> None:
    predictor.update(None, new_data)


def get_basis(predictor: combo.base_predictor, X: np.ndarray) -> np.ndarray:
    return predictor.get_basis(X)


def init_test(predictor: combo.base_predictor, test_X: np.ndarray) -> combo.variable:
    return combo.variable(X=test_X, Z=predictor.get_basis(test_X))


def get_score(predictor: combo.base_predictor, test: combo.variable, score: str) -> np.ndarray:
    if score == 'EI':
        return combo.search.score.EI(predictor, None, test)
    elif score == 'PI':
        return combo.search.score.PI(predictor, None, test)
    elif score == 'TS':
        return combo.search.score.TS(predictor, None, test, predictor.config.search.alpha)
    else:
        raise NotImplementedError('mode must be EI, PI or TS.')


class Centering:

    def __init__(self, X: np.ndarray, config: combo.misc.set_config) -> None:
        self.mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.std = np.where(std >= config.learning.epsilon, std, 1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class Predictor:

    def __init__(self, policy: 'Policy', num_rand_basis: int, config: combo.misc.set_config) -> None:
        self.policy = policy
        self.new_data = combo.variable()
        self.predictor = init_predictor(num_rand_basis=num_rand_basis, config=config)
        training = self.policy.training
        self.centering = Centering(training.X, config=config)
        learn(self.predictor, combo.variable(X=self.centering(training.X), t=training.t), num_rand_basis=num_rand_basis)

    def write(self, X: np.ndarray, test: combo.variable, t: np.ndarray, is_disp: bool = True) -> None:
        self.new_data.add(X=test.X, t=t, Z=test.Z)
        self.policy.write(X, t, is_disp)

    def init_test(self, X: np.ndarray) -> combo.variable:
        return init_test(self.predictor, self.centering(X))

    def get_score(self, test: combo.variable, score: str) -> np.ndarray:
        update(self.predictor, self.new_data)
        self.new_data = combo.variable()
        return get_score(self.predictor, test, score)


class Policy:

    def __init__(self, config: Optional[combo.misc.set_config] = None) -> None:
        super().__init__()
        self.training = combo.variable()
        self.history = combo.search.discrete.history()
        self.config = init_config(config)

    def write(self, X: np.ndarray, t: np.ndarray, is_disp: bool = True) -> None:
        st = self.history.total_num_search
        self.history.write(t, np.arange(st, st + t.shape[0]))
        self.training.add(X=X, t=t)
        if is_disp:
            combo.search.utility.show_search_results(self.history, t.shape[0])

    def learn(self, num_rand_basis: int) -> Predictor:
        return Predictor(self, num_rand_basis=num_rand_basis, config=self.config)


class Test:

    def __init__(self, X: np.ndarray) -> None:
        self.X = X

    def random_search(self, policy: Policy) -> np.ndarray:
        ...

    def bayes_search(self, predictor: Predictor) -> Tuple[np.ndarray, combo.variable]:
        ...
