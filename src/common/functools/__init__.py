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
import functools
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, NamedTuple, Optional, Type, TypeVar, cast


_T = TypeVar('_T')
_InstanceT = TypeVar('_InstanceT')
_ValueT = TypeVar('_ValueT')


if TYPE_CHECKING:
    from src.common.typing import SupportsGet
    from functools import _CacheInfo
    from typing import Protocol
    from src.common.typing import any, declval


    class property(Protocol[_InstanceT, _ValueT]):

        def __init__(self, fget: Optional[Callable[[_InstanceT], _ValueT]] = any,
                     fset: Optional[Callable[[_InstanceT, _ValueT], None]] = any,
                     fdel: Optional[Callable[[_InstanceT], None]] = any,
                     doc: Optional[str] = any) -> None: ...

        def getter(self, fget: Callable[[_InstanceT], _ValueT]) -> 'property[_InstanceT, _ValueT]': ...

        def setter(self, fset: Callable[[_InstanceT, _ValueT], None]) -> 'property[_InstanceT, _ValueT]': ...

        def deleter(self, fdel: Callable[[_InstanceT], None]) -> 'property[_InstanceT, _ValueT]': ...

        def __get__(self, obj: Optional[_InstanceT], type: Optional[Type[_InstanceT]] = any) -> _ValueT: ...

        def __set__(self, obj: _InstanceT, value: _ValueT) -> None: ...

        def __delete__(self, obj: _InstanceT) -> None: ...

        @builtins.property
        def fget(self) -> Optional[Callable[[_InstanceT], _ValueT]]: ...

        @builtins.property
        def fset(self) -> Optional[Callable[[_InstanceT, _ValueT], None]]: ...

        @builtins.property
        def fdel(self) -> Optional[Callable[[_InstanceT], None]]: ...


    _AnyCallable = TypeVar('_AnyCallable', bound=Callable[..., Any])


    class _lru_cache_wrapper(Protocol[_AnyCallable]):

        @builtins.property
        def __wrapped__(self) -> _AnyCallable: ...

        __call__: _AnyCallable

        def cache_info(self) -> _CacheInfo: ...

        def cache_clear(self) -> None: ...


    class _lru_cache(Protocol):
        def __init__(self, maxsize: Optional[int] = any, typed: bool = any) -> None: ...
        def __call__(self, f: _AnyCallable) -> _AnyCallable: ...


lru_cache: 'Type[_lru_cache]' = cast('Type[_lru_cache]', functools.lru_cache)


_InstanceT_contra = TypeVar('_InstanceT_contra', contravariant=True)
_ValueT_co = TypeVar('_ValueT_co', covariant=True)


def cached_property(fget: Callable[[_InstanceT_contra], _ValueT_co]) -> 'SupportsGet[_ValueT_co]':
    return builtins.property(functools.lru_cache()(fget))
