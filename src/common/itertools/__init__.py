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


import builtins
import itertools
from typing import Any, Callable, Generic, Iterable, Iterator, Tuple, TypeVar


_T = TypeVar('_T')
_T1 = TypeVar('_T1')
_S = TypeVar('_S')


class CopiableIterator(Iterator[_T], Generic[_T]):

    def __init__(self, iterator: Iterator[_T]) -> None:
        self._iterator = iterator

    def __deepcopy__(self, memo: Any) -> 'CopiableIterator[_T]':
        iterator, = itertools.tee(self._iterator, 1)
        return CopiableIterator(iterator)

    def __iter__(self) -> Iterator[_T]:
        return self._iterator.__iter__()

    def __next__(self) -> _T:
        return self._iterator.__next__()


class Iterand(Iterator[_T], Generic[_T]):

    def map(self, function: Callable[[_T], _S]) -> 'Iterand[_S]':
        return Iterand(builtins.map(function, self._operand))

    def flat_map(self, function: Callable[[_T], Iterable[_S]]) -> 'Iterand[_S]':
        return Iterand(itertools.chain.from_iterable(builtins.map(function, self._operand)))

    def zip(self, iter1: Iterator[_T1], function: Callable[[_T, _T1], _S]) -> 'Iterand[_S]':
        return Iterand(builtins.map(function, self._operand, iter1))

    def enumerate(self, function: Callable[[int, _T], _S]) -> 'Iterand[_S]':
        return Iterand(builtins.map(function, itertools.count(), self._operand))

    def __init__(self, _operand: Iterator[_T]) -> None:
        self._operand = _operand

    def __iter__(self) -> 'Iterator[_T]':
        return self._operand

    def __next__(self) -> '_T':
        return self._operand.__next__()
