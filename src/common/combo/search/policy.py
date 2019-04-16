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


# from copy import copy, deepcopy
# from typing import Callable, Generator, Optional, Union
#
# import combo
# import numpy as np
# import pandas as pd
#
# from src.common.genertools.generand import Generand
# from .training import Training
#
#
# def write(new_data: combo.variable, test: combo.variable, t: np.ndarray) -> None:
#     new_data.add(X=test.X, t=t, Z=test.Z)
#
#
# def get_random_action(self: combo.search.discrete.policy, test: combo.variable, num_search_each_probe: int = 1, is_disp: bool = True) -> np.ndarray:
#     other = copy(self)
#     other.test = test
#     N = num_search_each_probe
#     if is_disp and N > 1:
#         combo.search.utility.show_start_message_multi_search(other.history.num_runs)
#     action = other.get_random_action(N)
#     return action
#
#
# def random_search(self: combo.search.discrete.policy, test: combo.variable, max_num_probes: int = 1, num_search_each_probe: int = 1, simulator: Optional[Callable[[np.ndarray], np.ndarray]] = None, is_disp: bool = True):
#     N = num_search_each_probe
#     if is_disp:
#         combo.search.utility.show_interactive_mode(simulator, self.history)
#     for n in range(max_num_probes):
#         action = get_random_action(self, test, N, is_disp)
#         if simulator is None:
#             return action
#         t = simulator(action)
#         write(self, test[action, :], t)
#     return deepcopy(self.history)
#
#
# def learn(predictor: combo.base_predictor, training: combo.variable, new_data: combo.variable, num_rand_basis: int) -> None:
#     training.add(X=new_data.X, t=new_data.t, Z=new_data.Z)
#     new_data.X = None
#     new_data.t = None
#     new_data.Z = None
#     predictor.fit(training, num_rand_basis)
#     training.Z = predictor.get_basis(training.X)
#     predictor.prepare(training)
#
#
# def update(predictor: combo.base_predictor, new_data: combo.variable) -> None:
#     predictor.update(None, new_data)
#
#
# def get_bayes_action(self: combo.search.discrete.policy, test: combo.variable, score: str = 'TS', num_search_each_probe: int = 1, is_disp: bool = True) -> np.ndarray:
#     other = copy(self)
#     test.Z = other.predictor.get_basis(test.X)
#     other.test = test
#     N = num_search_each_probe
#     if is_disp and N > 1:
#         combo.search.utility.show_start_message_multi_search(other.history.num_runs, score)
#     K = other.config.search.multi_probe_num_sampling
#     alpha = other.config.search.alpha
#     action = other.get_actions(score, N, K, alpha)
#     return action
#
#
# def bayes_search(self: combo.search.discrete.policy, test: combo.variable, max_num_probes: int = 1, num_search_each_probe: int = 1, is_disp: bool = True, simulator: Optional[Callable[[np.ndarray], np.ndarray]] = None, score: str = 'TS', interval: int = 0, num_rand_basis: int = 0) -> Union[np.ndarray, combo.search.discrete.history]:
#     if is_disp:
#         combo.search.utility.show_interactive_mode(simulator, self.history)
#     N = num_search_each_probe
#     for n in range(max_num_probes):
#         if combo.search.utility.is_learning(n, interval):
#             learn(self, num_rand_basis)
#         else:
#             update(self)
#         action = get_bayes_action(self, test, score, N, is_disp=is_disp)
#         if simulator is None:
#             return action
#         t = simulator(action)
#         write(self, test[action, :], t)
#     return deepcopy(self.history)
#
#
# class Policy:
#
#     def __init__(self, config: Optional[combo.misc.set_config] = None) -> None:
#         super().__init__()
#         self._X = pd.DataFrame()
#         self._policy = combo.search.discrete.policy(test_X=combo.variable(), config=config)
