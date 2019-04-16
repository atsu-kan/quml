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


from typing import Callable, Generic, Iterator, Optional, TypeVar

import numpy as np

from src.pylabzmqinterface.adaptee import Adaptee


_T = TypeVar('_T')


class Connection(Adaptee, Generic[_T]):
    
    def __init__(self, on_connection: Iterator[_T], create_adaptee: Callable[[_T], Adaptee]) -> None:
        super().__init__()
        self._adaptee: Optional[Adaptee] = None
        self._create_adaptee = create_adaptee
        self._on_connection = on_connection

    def write(self, param: np.ndarray, result: np.ndarray):
        if self._adaptee is not None:
            self._adaptee.write(param, result)

    def create_reader(self) -> Iterator[np.ndarray]:
        for on_session in self._on_connection:
            self._adaptee = self._create_adaptee(on_session)
            yield from self._adaptee.reader
