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


# from abc import abstractmethod
# from typing import Generic, Iterator, TypeVar
#
#
# _T_co = TypeVar('_T_co', covariant=True)
#
#
# class IteratorBase(Iterator[_T_co], Generic[_T_co]):
#
#     @abstractmethod
#     def __iter__(self) -> Iterator[_T_co]:
#         ...
#
#     def __next__(self) -> _T_co:
#         return self.__iter__().__next__()
