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


from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, GenericMeta, TypeVar


_T = TypeVar('_T')
_ArgT = TypeVar('_ArgT', contravariant=True)
_ReturnT = TypeVar('_ReturnT', covariant=True)
_SubT = TypeVar('_SubT', bound='Operator')


if TYPE_CHECKING:
    from typing import Protocol


    class OperatorProtocol(Protocol[_ArgT, _ReturnT]):

        def __ror__(self, opearnd: _ArgT) -> _ReturnT:
            ...


class Operator(Generic[_ArgT, _T]):

    @abstractmethod
    def __ror__(self, operand: _ArgT) -> _T:
        ...
    #
    # def __or__(self: 'OperatorProtocol[_ArgT, _T]', other: 'OperatorProtocol[_T, _ReturnT]') -> 'OperatorProtocol[_ArgT, _ReturnT]':
    #     return CompositeOperator(self, other)


class CompositeOperator(Generic[_ArgT, _T, _ReturnT]):

    def __init__(self, a: 'OperatorProtocol[_ArgT, _T]', b: 'OperatorProtocol[_T, _ReturnT]') -> None:
        self._a = a
        self._b = b

    def __ror__(self, operand: _ArgT) -> _ReturnT:
        return operand | self._a | self._b
