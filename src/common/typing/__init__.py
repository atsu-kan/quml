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


import typing
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, overload


_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)


def just(obj: Optional[_T]) -> _T:
    return typing.cast(_T, obj)


class _declval:

    def __getitem__(self, tye: Type[_T]) -> _T:
        raise RuntimeError()


declval = _declval()


any: Any
#
#
# def cast(typ: Type[_T]) -> Callable[[Any], _T]:
#
#     def wrapper(val: Any) -> _T:
#         return val
#
#     return wrapper
#
#
# class Cast(Generic[_T]):
#
#     def __call__(self, val: Any) -> _T:
#         return val


if TYPE_CHECKING:
    from typing import Protocol


    _InstanceT_contra = TypeVar('_InstanceT_contra', contravariant=True)
    _ValueT_co = TypeVar('_ValueT_co', covariant=True)


    class SupportsGet(Protocol[_ValueT_co]):

        def __get__(self, instance: Optional[_InstanceT_contra], owner: Optional[Type[_InstanceT_contra]]) -> _ValueT_co:
            ...
