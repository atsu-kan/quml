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
from functools import wraps
import queue
from queue import Queue
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import numpy as np

from src.pylabzmqinterface.adaptee import Adaptee


_CallableT = TypeVar('_CallableT', bound=Callable[..., Any])


def _throwable(wrapped: _CallableT) -> _CallableT:
    @wraps(wrapped)
    def wrapper(self: 'Adapter', *args: Any, **kwargs: Any) -> Any:
        if self._future.done():
            try:
                self._future.result()
            except:
                raise
        return wrapped(self, *args, **kwargs)

    return cast(_CallableT, wrapper)


class Adapter:

    def __init__(self, adaptee: Adaptee) -> None:
        super().__init__()
        self._adaptee = adaptee
        self._next_queue: Queue[Optional[np.ndarray]] = Queue()
        self._last_queue: Queue[Callable[[], bool]] = Queue()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = self._executor.submit(self._run)
        self._submit_to_read()

    @_throwable
    def write(self, last_param: np.ndarray, last_result: np.ndarray) -> None:
        self._submit_to_write(last_param, last_result)

    @_throwable
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            next_param = self._next_queue.get_nowait()
            self._submit_to_read()
            return next_param is not None, next_param
        except queue.Empty:
            return True, None

    @_throwable
    def shutdown(self) -> None:
        self._last_queue.put_nowait(lambda: False)

    def _submit_to_read(self) -> None:

        def request() -> bool:
            next_param = next(self._adaptee.reader, None)
            self._next_queue.put_nowait(next_param)
            return next_param is not None

        self._last_queue.put_nowait(request)

    def _submit_to_write(self, last_param: np.ndarray, last_result: np.ndarray) -> None:

        def request() -> bool:
            self._adaptee.write(last_param, last_result)
            return True

        self._last_queue.put_nowait(request)

    def _run(self) -> None:
        while True:
            if not self._last_queue.get()():
                break
