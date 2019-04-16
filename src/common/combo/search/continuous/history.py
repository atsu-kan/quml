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


from typing import Any, Tuple

from combo.search import utility
import numpy as np


MAX_SEARCH = int(30000)


class History:

    def __init__(self) -> None:
        self.num_runs = int(0)
        self.total_num_search = int(0)
        self.fx = np.zeros(MAX_SEARCH, dtype=float)
        self.choosed_X = np.full(MAX_SEARCH, None)
        self.terminal_num_run = np.zeros(MAX_SEARCH, dtype=int)

    def write(self, t: np.ndarray, X: np.ndarray) -> None:
        N = utility.length_vector(t)
        st = self.total_num_search
        en = st + N

        self.terminal_num_run[self.num_runs] = en
        self.fx[st:en] = t
        self.choosed_X[st:en] = list(X)
        self.num_runs += 1
        self.total_num_search += N

    def export_sequence_best_fx(self) -> Tuple[np.ndarray, np.ndarray]:
        best_fx = np.zeros(self.num_runs)
        best_X = np.full(self.num_runs, None)
        for n in range(self.num_runs):
            index = np.argmax(self.fx[0:self.terminal_num_run[n]])
            best_X[n] = self.choosed_X[index]
            best_fx[n] = self.fx[index]

        return best_fx[self.num_runs], np.array(list(best_X[0:self.num_runs]))

    def export_all_sequence_best_fx(self) -> Tuple[np.ndarray, np.ndarray]:
        best_fx = np.zeros(self.total_num_search)
        best_X = np.zeros(self.total_num_search, None)
        best_fx[0] = self.fx[0]
        best_X[0] = self.choosed_X[0]

        for n in range(1, self.total_num_search):
            if best_fx[n-1] < self.fx[n]:
                best_fx[n] = self.fx[n]
                best_X[n] = self.choosed_X[n]
            else:
                best_fx[n] = best_fx[n-1]
                best_X[n] = best_X[n-1]

        return best_fx[self.total_num_search], np.array(list(best_X[0: self.total_num_search]))

    def save(self, filename: Any) -> None:
        N = self.total_num_search
        M = self.num_runs
        np.savez_compressed(filename, num_runs=M, total_num_search=N,
                            fx=self.fx[0:N],
                            choosed_X=self.choosed_X[0:N],
                            terminal_num_run=self.terminal_num_run[0:M])

    def load(self, filename: Any) -> None:
        data = np.load(filename)
        M = data['num_runs']
        N = data['total_num_search']
        self.num_runs = M
        self.total_num_search = N
        self.fx[0:N] = data['fx']
        self.choosed_X[0:N] = data['choosed_X']
        self.terminal_num_run[0:M] = data['terminal_num_run']
