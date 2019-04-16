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


from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
from queue import Queue
from types import TracebackType
from typing import Any, IO, Optional, Type, TypeVar, cast

from sklearn.externals import joblib


_T = TypeVar('_T')
#
#
# class _Stream:
#
#     def __init__(self):
#         self._queue: Queue[bytes] = Queue(maxsize=1)
#         self._buffer = b''
#
#     def write(self, s: bytes) -> int:
#         self._queue.put(s)
#         return len(s)
#
#     def flush(self) -> None:
#         pass
#
#     def read(self, n: int) -> bytes:
#         while len(self._buffer) < n:
#             self._buffer += self._queue.get()
#         s = self._buffer[:n]
#         self._buffer = self._buffer[n:]
#         return s
#     #
#     # def __enter__(self) -> '_Stream':
#     #     self._bytes_queue.mutex.__enter__()
#     #     self._bytes_queue.not_empty.__enter__()
#     #     self._bytes_queue.not_full.__enter__()
#     #     self._bytes_queue.all_tasks_done.__enter__()
#     #     self._bytes_stream.__enter__()
#     #     return self
#     #
#     # def __exit__(self, *args: Any) -> None:
#     #     self._bytes_stream.__exit__(*args)
#     #     self._bytes_queue.all_tasks_done.__exit__(*args)
#     #     self._bytes_queue.not_full.__exit__(*args)
#     #     self._bytes_queue.not_empty.__exit__(*args)
#     #     self._bytes_queue.mutex.__exit__(*args)


def deepcopy(x: _T) -> _T:
    stream = BytesIO()
    joblib.dump(x, stream)
    return joblib.load(stream)
