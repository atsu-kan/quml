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


from typing import Callable, Generator, Generic, Iterable, Optional, Tuple, Type, TypeVar
from types import TracebackType

from src.common import genertools
from src.common.itertools.iterand import Iterand


_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_U = TypeVar('_U')
_ArgT = TypeVar('_ArgT')
_YieldT = TypeVar('_YieldT')
_YieldT_co = TypeVar('_YieldT_co', covariant=True)
_SendT = TypeVar('_SendT')
_SendT_contra = TypeVar('_SendT_contra', contravariant=True)
_ReturnT = TypeVar('_ReturnT')
_ReturnT_co = TypeVar('_ReturnT_co', covariant=True)
_OtherReturnT_co = TypeVar('_OtherReturnT_co', covariant=True)
_ExceptionT = TypeVar('_ExceptionT', bound=BaseException)


class Generand(Generator[_ArgT, _ReturnT, None], Generic[_ArgT, _ReturnT]):

    def map(self, function: Callable[[_ArgT], Generator[_YieldT, _SendT, _ReturnT]]) -> 'Generand[_YieldT, _SendT]':
        return Generand(genertools._map(function, self._generator))

    def map_yield(self, function: Callable[[_ArgT], _YieldT]) -> 'Generand[_YieldT, _ReturnT]':
        return Generand(genertools._map_yield(function, self._generator))

    def map_return(self, function: Callable[[_SendT], _ReturnT]) -> 'Generand[_ArgT, _SendT]':
        return Generand(genertools._map_return(function, self._generator))

    def flat_map(self, function: Callable[[_ArgT], Generator[Iterable[_YieldT], Iterable[_SendT], _ReturnT]]) -> 'Iterand[Generand[_YieldT, _SendT]]':
        return Iterand(genertools._flat_map(function, self._generator)).map(Generand)

    def eval(self) -> Tuple[_ArgT, Callable[[_ReturnT], None]]:
        return genertools.call(self._generator)

    def __init__(self, generator: Generator[_ArgT, _ReturnT, None]) -> None:
        self._generator = generator

    def __iter__(self) -> Generator[_ArgT, _ReturnT, None]:
        return self._generator.__iter__()

    def __next__(self) -> _ArgT:
        return self._generator.__next__()

    def close(self) -> None:
        return self._generator.close()

    def send(self, value: _ReturnT) -> _ArgT:
        return self._generator.send(value)

    def throw(self, type: Type[_ExceptionT], value: Optional[_ExceptionT] = None, traceback: Optional[TracebackType] = None) -> _ArgT:
        return self._generator.throw(type, value, traceback)
