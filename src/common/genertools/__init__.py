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
from typing import Any, Callable, Dict, Generator, Generic, Iterable, Iterator, Optional, Tuple, Type, TypeVar
from types import TracebackType


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


def call(generator: Generator[_YieldT_co, _SendT_contra, _ReturnT_co]) -> Tuple[_YieldT_co, Callable[[_SendT_contra], _ReturnT_co]]:
    def send(value: _SendT_contra) -> _ReturnT_co:
        try:
            generator.send(value)
            generator.close()
            raise RuntimeError()
        except StopIteration as e:
            return e.value
    try:
        return builtins.next(generator), send
    except StopIteration as e:
        raise RuntimeError() from e


def _map(function: Callable[[_ArgT], Generator[_YieldT_co, _SendT_contra, _ReturnT]], generator: Generator[_ArgT, _ReturnT, _OtherReturnT_co]) -> Generator[_YieldT_co, _SendT_contra, _OtherReturnT_co]:
    outer_value, outer_send = call(generator)
    inner_value, inner_send = call(function(outer_value))
    return outer_send(inner_send((yield inner_value)))


def _map_yield(function: Callable[[_ArgT], _YieldT], generator: Generator[_ArgT, _ReturnT, _OtherReturnT_co]) -> Generator[_YieldT, _ReturnT, _OtherReturnT_co]:
    _value, _send = call(generator)
    return _send((yield function(_value)))


def _map_return(function: Callable[[_SendT], _ReturnT], generator: Generator[_ArgT, _ReturnT, _OtherReturnT_co]) -> Generator[_ArgT, _SendT, _OtherReturnT_co]:
    _value, _send = call(generator)
    return _send(function((yield _value)))


def _flatten(generator: Generator[Iterable[_YieldT], Iterable[_SendT_contra], None]) -> Iterator[Generator[_YieldT, _SendT_contra, None]]:
    _value, _send = call(generator)
    args = list(_value)
    results: Dict[int, _SendT_contra] = {}

    def each_arg(i: int, arg: _YieldT) -> Generator[_YieldT, _SendT_contra, None]:
        results[i] = (yield arg)
        if not (len(results) < len(args)):
            _send([results[key] for key in sorted(results.keys())])

    yield from (each_arg(i, arg) for i, arg in enumerate(args))


def _flat_map(function: Callable[[_ArgT], Generator[Iterable[_YieldT], Iterable[_SendT], _ReturnT]], generator: Generator[_ArgT, _ReturnT, None]) -> Iterator[Generator[_YieldT, _SendT, None]]:
    return _flatten(_map(function, generator))


class Generand(Iterator[Generator[_ArgT, _ReturnT, None]], Generic[_ArgT, _ReturnT]):

    def __init__(self, iterator: Iterator[Generator[_ArgT, _ReturnT, None]]) -> None:
        self._iterator = iterator

    def map(self, function: Callable[[_ArgT], Generator[_YieldT, _SendT, _ReturnT]]) -> 'Generand[_YieldT, _SendT]':
        return Generand((_map(function, generator) for generator in self._iterator))

    def map_yield(self, function: Callable[[_ArgT], _YieldT]) -> 'Generand[_YieldT, _ReturnT]':
        return Generand((_map_yield(function, generator) for generator in self._iterator))

    def map_return(self, function: Callable[[_SendT], _ReturnT]) -> 'Generand[_ArgT, _SendT]':
        return Generand((_map_return(function, generator) for generator in self._iterator))

    def flat_map(self, function: Callable[[_ArgT], Generator[Iterable[_YieldT], Iterable[_SendT], _ReturnT]]) -> 'Generand[_YieldT, _SendT]':
        return Generand(itertools.chain.from_iterable((_flat_map(function, generator) for generator in self._iterator)))

    def __iter__(self) -> Iterator[Generator[_ArgT, _ReturnT, None]]:
        return self._iterator.__iter__()

    def __next__(self) -> Generator[_ArgT, _ReturnT, None]:
        return self._iterator.__next__()


def from_generator(generator: Generator[_ArgT, _ReturnT, None]) -> Generand[_ArgT, _ReturnT]:
    return Generand(iter([generator]))
