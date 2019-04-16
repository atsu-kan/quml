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


import itertools
from typing import Iterator


class Counter(Iterator[int]):

    def __init__(self, start: int = 0, step: int = 1) -> None:
        self._iterator = itertools.count(start, step)

    def __iter__(self) -> 'Counter':
        return self

    def __next__(self) -> int:
        return self._iterator.__next__()
