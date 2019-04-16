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
from typing import Callable, Generic, Iterable, Tuple, TypeVar


_T = TypeVar('_T')
_AT = TypeVar('_AT')
_BT = TypeVar('_BT')
_ArgT = TypeVar('_ArgT')
_AArgT = TypeVar('_AArgT')
_BArgT = TypeVar('_BArgT')
_ReturnT = TypeVar('_ReturnT')
_ReturnAT = TypeVar('_ReturnAT')
_ReturnBT = TypeVar('_ReturnBT')


class FlatMap(Generic[_ArgT, _ReturnT]):

    def __init__(self, function: Callable[[_ArgT], Iterable[_ReturnT]]) -> None:
        super().__init__()
        self._function = function

    def __ror__(self, operand: Iterable[_ArgT]) -> Iterable[_ReturnT]:
        return operand | Map(self._function) | flatten


class _Flatten:

    def __ror__(self, operand: Iterable[Iterable[_T]]) -> Iterable[_T]:
        return itertools.chain.from_iterable(operand)


flatten = _Flatten()


class Map(Generic[_ArgT, _ReturnT]):

    def __init__(self, function: Callable[[_ArgT], _ReturnT]) -> None:
        super().__init__()
        self._function = function

    def __ror__(self, operand: Iterable[_ArgT]) -> Iterable[_ReturnT]:
        return builtins.map(self._function, operand)


class Zip(Generic[_AArgT, _BArgT, _ReturnT]):
    
    def __init__(self, function: Callable[[_AArgT, _BArgT], _ReturnT]) -> None:
        self._function = function
    
    def __ror__(self, operand: Tuple[Iterable[_AArgT], Iterable[_BArgT]]) -> Iterable[_ReturnT]:
        return builtins.map(self._function, *operand)


class _Zip:

    def __ror__(self, operand: Tuple[Iterable[_AT], Iterable[_BT]]) -> Iterable[Tuple[_AT, _BT]]:
        return builtins.zip(*operand)


zip = _Zip()
