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
from typing import Iterator

import numpy as np


class Adaptee(ABC):

    @abstractmethod
    def create_reader(self) -> Iterator[np.ndarray]:
        ...

    @abstractmethod
    def write(self, param: np.ndarray, result: np.ndarray) -> None:
        ...

    def __init__(self) -> None:
        self._reader = self.create_reader()

    @property
    def reader(self) -> Iterator[np.ndarray]:
        return self._reader
