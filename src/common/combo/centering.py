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


import numpy as np


class Centering:

    def __init__(self, X: np.ndarray, epsilon: float) -> None:
        self.mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.std = np.where(std >= epsilon, std, 1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
