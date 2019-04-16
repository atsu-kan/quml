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


from copy import deepcopy
import pickle
from typing import Any, Callable, Optional, Union, cast, overload

from combo import base_predictor as Predictor
from combo.blm import predictor as blm_predictor
from combo.gp import predictor as gp_predictor
from combo.misc import set_config as Config
from combo.search import utility
import combo.search.score
from combo.variable import variable as Variable
import numpy as np

from .action import Action
from .history import History
from .chooser import Chooser
from .types import Simulator


class Policy:

    def __init__(self, chooser: Chooser, config: Optional[Config] = None) -> None:
        self.predictor: Optional[Predictor] = None
        self.training = Variable()
        self.chooser = chooser
        self.history = History()
        self.config = self._set_config(config)
        self.new_data = Variable()

    @property
    def training_len(self) -> int:
        if self.training.t is None:
            return 0
        else:
            return self.training.t.shape[0]

    def write(self, action: Action, t: np.ndarray, *, is_disp: bool = True) -> None:
        self.new_data.add(X=action.test.X, t=t, Z=action.test.Z)
        self.history.write(t, action.test.X)
        self.training.add(X=action.test.X, t=t, Z=action.test.Z)

        if is_disp:
            utility.show_search_results(self.history, t.shape[0])

    @overload
    def random_search(
            self, *,
            num_search_each_probe: int = 1,
            is_disp: bool = True
    ) -> Action: ...

    @overload
    def random_search(
            self, *,
            num_search_each_probe: int = 1,
            is_disp: bool = True,
            max_num_probes: int,
            simulator: Simulator
    ) -> History: ...

    def random_search(
            self, *,
            num_search_each_probe: int = 1,
            is_disp: bool = True,
            max_num_probes: Optional[int] = None,
            simulator: Optional[Simulator] = None
    ) -> Union[Action, History]:

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        N = int(num_search_each_probe)

        if is_disp:
            utility.show_interactive_mode(simulator, self.history)

        for n in range(0, max_num_probes):

            if is_disp and N > 1:
                utility.show_start_message_multi_search(self.history.num_runs)

            action = self.get_random_action(N)

            if simulator is None:
                return action

            t = simulator(action.X)

            self.write(action, t, is_disp=is_disp)

        return deepcopy(self.history)

    @overload
    def bayes_search(
            self, *,
            training: Optional[Variable] = None,
            num_search_each_probe: int = 1,
            predictor: Optional[Predictor] = None,
            is_disp: bool = True,
            score: str = 'TS',
            interval: int = 0,
            num_rand_basis: int = 0
    ) -> Action: ...

    @overload
    def bayes_search(
            self, *,
            training: Optional[Variable] = None,
            num_search_each_probe: int = 1,
            predictor: Optional[Predictor] = None,
            is_disp: bool = True,
            score: str = 'TS',
            interval: int = 0,
            num_rand_basis: int = 0,
            max_num_probes: int,
            simulator: Simulator
    ) -> History: ...

    def bayes_search(
            self, *,
            training: Optional[Variable] = None,
            num_search_each_probe: int = 1,
            predictor: Optional[Predictor] = None,
            is_disp: bool = True,
            score: str = 'TS',
            interval: int = 0,
            num_rand_basis: int = 0,
            max_num_probes: Optional[int] = None,
            simulator: Optional[Simulator] = None
    ) -> Union[Action, History]:

        if max_num_probes is None:
            max_num_probes = 1
            simulator = None

        is_rand_expans = False if num_rand_basis == 0 else True

        self.training = self._set_training(training)

        if predictor is None:
            self.predictor = self._init_predictor(is_rand_expans)
        else:
            self.predictor = predictor

        N = int(num_search_each_probe)

        for n in range(max_num_probes):

            if utility.is_learning(n, interval):
                self.predictor.fit(self.training, num_rand_basis)
                self.training.Z = self.predictor.get_basis(self.training.X)
                self.predictor.prepare(self.training)
            else:
                try:
                    if self.new_data.Z is not None:
                        self.predictor.update(self.training, self.new_data)
                except:
                    self.predictor.prepare(self.training)
            self.new_data = Variable()

            if num_search_each_probe != 1:
                utility.show_start_message_multi_search(self.history.num_runs,
                                                        score)

            K = self.config.search.multi_probe_num_sampling
            alpha = self.config.search.alpha
            action = self.get_actions(score, N, K, alpha)

            if simulator is None:
                return action

            t = simulator(action.X)

            self.write(action, t, is_disp=is_disp)

        return deepcopy(self.history)

    def get_score(self, mode: str, predictor: Optional[Predictor] = None, training: Optional[Variable] = None, alpha: float = 1) -> Action:
        self._set_training(training)
        self._set_predictor(predictor)

        f: Callable[[Variable], np.ndarray]
        if mode == 'EI':
            f = lambda test: combo.search.score.EI(predictor, training, test)
        elif mode == 'PI':
            f = lambda test: combo.search.score.PI(predictor, training, test)
        elif mode == 'TS':
            f = lambda test: combo.search.score.TS(predictor, training, test, alpha)
        else:
            raise NotImplementedError('mode must be EI, PI or TS.')

        def get_test(test_X: np.ndarray) -> Variable:
            test = Variable(X=test_X, Z=cast(Predictor, predictor).get_basis(test_X))
            test.t = f(test)
            return test

        return self.chooser.choose_bayes_action(get_test)

    def get_marginal_score(self, mode: str, choosed_actions: Action, N: int, alpha: float) -> Action:
        f = np.full(1, None)
        new_test = Variable(X=choosed_actions.test.X, Z=choosed_actions.test.Z)
        virtual_t \
            = cast(Predictor, self.predictor).get_predict_samples(self.training, new_test, 1)

        for n in range(f.shape[0]):
            predictor = deepcopy(cast(Predictor, self.predictor))
            train = deepcopy(self.training)
            virtual_train = Variable(X=new_test.X, t=virtual_t, Z=new_test.Z)

            train.add(virtual_train.X, virtual_train.t, virtual_train.Z)

            try:
                predictor.update(train, virtual_train)
            except:
                predictor.prepare(train)

            f[n] = self.get_score(mode, predictor, train)
        return f.item()

    def get_actions(self, mode: str, N: int, K: int, alpha: float) -> Action:
        choosed_actions = self.get_score(mode, self.predictor, self.training, alpha)

        for n in range(1, N):
            choosed_actions.add(self.get_marginal_score(mode, choosed_actions, K, alpha))

        return choosed_actions

    def get_random_action(self, N: int) -> Action:
        return self.chooser.choose_random_actions(N)

    def load(self, file_history: Any, file_training: Any = None, file_predictor: Any = None) -> None:
        self.history.load(file_history)

        if file_training is None:
            N = self.history.total_num_search
            X = self.history.choosed_X[0:N]
            t = self.history.fx[0:N]
            self.training = Variable(X=X, t=t)
        else:
            self.training = Variable()
            self.training.load(file_training)

        if file_predictor is not None:
            with open(file_predictor, 'rb') as f:
                self.predictor = pickle.load(f)

    def export_predictor(self) -> Optional[Predictor]:
        return self.predictor

    def export_training(self) -> Variable:
        return self.training

    def export_history(self) -> History:
        return self.history

    def _set_predictor(self, predictor: Optional[Predictor] = None) -> Optional[Predictor]:
        if predictor is None:
            predictor = self.predictor
        return predictor

    def _init_predictor(self, is_rand_expans: bool, predictor: Optional[Predictor] = None) -> Predictor:
        self.predictor = self._set_predictor(predictor)
        if self.predictor is None:
            if is_rand_expans:
                self.predictor = blm_predictor(self.config)
            else:
                self.predictor = gp_predictor(self.config)

        return self.predictor

    def _set_training(self, training: Optional[Variable] = None) -> Variable:
        if training is None:
            training = self.training
        return training

    def _set_config(self, config: Optional[Config] = None) -> Config:
        if config is None:
            config = Config()
        return config
