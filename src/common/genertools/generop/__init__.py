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
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Generic, Iterable, Iterator, Tuple, TypeVar


_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_U = TypeVar('_U')
_ArgT = TypeVar('_ArgT')
_ArgT_contra = TypeVar('_ArgT_contra')
_YieldT = TypeVar('_YieldT')
_YieldU = TypeVar('_YieldU')
_YieldT_co = TypeVar('_YieldT_co', covariant=True)
_SendT = TypeVar('_SendT')
_SendU = TypeVar('_SendU')
_SendT_contra = TypeVar('_SendT_contra', contravariant=True)
_ReturnT = TypeVar('_ReturnT')
_ReturnT_co = TypeVar('_ReturnT_co', covariant=True)
_OtherReturnT = TypeVar('_OtherReturnT')


def _next(iterator: Iterator[_T_co]) -> _T_co:
    try:
        return builtins.next(iterator)
    except StopIteration as e:
        raise RuntimeError() from e


def _send(generator: Generator[_YieldT_co, _SendT_contra, _ReturnT_co], value: _SendT_contra) -> _ReturnT_co:
    try:
        generator.send(value)
    except StopIteration as e:
        return e.value
    finally:
        generator.close()
    raise RuntimeError()


class FlatMap(Generic[_ArgT, _YieldT, _SendT, _ReturnT]):

    def __init__(self, function: Callable[[_ArgT], Generator[Iterable[_YieldT], Iterable[_SendT], _ReturnT]]) -> None:
        self._function = function

    def __ror__(self, operand: Generator[_ArgT, _ReturnT, None]) -> Iterator[Generator[_YieldT, _SendT, None]]:
        return operand | Map(self._function) | flatten


class Flatten(Generic[_YieldT, _SendT]):

    def __ror__(self, operand: Generator[Iterable[_YieldT], Iterable[_SendT], None]) -> Iterator[Generator[_YieldT, _SendT, None]]:
        args, send = operand | unpack
        arg_list = list(args)
        result_dict: Dict[int, _SendT] = {}

        def each_arg(i: int, arg: Any) -> Generator[_YieldT, _SendT, None]:
            result_dict[i] = (yield arg)
            if not (len(result_dict) < len(arg_list)):
                send([result_dict[key] for key in sorted(result_dict.keys())])

        yield from (each_arg(i, arg) for i, arg in enumerate(arg_list))


class _Flatten:

    def __ror__(self, operand: Generator[Iterable[_YieldT], Iterable[_SendT], None]) -> Iterator[Generator[_YieldT, _SendT, None]]:
        return operand | Flatten[_YieldT, _SendT]()


flatten = _Flatten()


class Map(Generic[_ArgT, _YieldT, _SendT, _ReturnT]):

    def __init__(self, function: Callable[[_ArgT], Generator[_YieldT, _SendT, _ReturnT]]) -> None:
        self._function = function

    def __ror__(self, operand: Generator[_ArgT, _ReturnT, None]) -> Generator[_YieldT, _SendT, None]:
        arg_, return_ = operand | unpack
        yield_, send_ = self._function(arg_) | unpack
        return_(send_((yield yield_)))


class MapYield(Generic[_ArgT, _YieldT]):

    def __init__(self, function: Callable[[_ArgT], _YieldT]) -> None:
        self._function = function

    def __ror__(self, operand: Generator[_ArgT, _T, None]) -> Generator[_YieldT, _T, None]:
        arg_, return_ = operand | unpack
        return_((yield self._function(arg_)))


class MapReturn(Generic[_SendT, _ReturnT]):

    def __init__(self, function: Callable[[_SendT], _ReturnT]) -> None:
        self._function = function

    def __ror__(self, operand: Generator[_T, _ReturnT, None]) -> Generator[_T, _SendT, None]:
        arg_, return_ = operand | unpack
        return_(self._function((yield arg_)))


class Unpack(Generic[_YieldT, _SendT, _ReturnT]):

    def __ror__(self, operand: Generator[_YieldT, _SendT, _ReturnT]) -> Tuple[_YieldT, Callable[[_SendT], _ReturnT]]:
        return _next(operand), lambda value: _send(operand, value)


class _Unpack:

    def __ror__(self, operand: Generator[_YieldT, _SendT, _ReturnT]) -> Tuple[_YieldT, Callable[[_SendT], _ReturnT]]:
        return operand | Unpack[_YieldT, _SendT, _ReturnT]()


unpack = _Unpack()
