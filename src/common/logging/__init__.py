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
from collections import OrderedDict
from contextlib import ContextDecorator, contextmanager
from copy import copy
import inspect
import json
import logging
from logging import Handler, StreamHandler
import logging.config
import sys
import threading
from time import gmtime, localtime, strftime
from threading import RLock
from typing import TYPE_CHECKING, Any, AnyStr, Callable, ContextManager, Generic, IO, Iterator, List, Optional, TextIO, TypeVar, cast

from IPython.display import DisplayObject, Pretty, TextDisplayObject, display, display_pretty
from IPython.utils.io import Tee

from src.common.os import pushd


_AnyStr_contra = TypeVar('_AnyStr_contra', bytes, str, contravariant=True)


if TYPE_CHECKING:
    from typing import Protocol

    class Writer(Protocol[_AnyStr_contra]):

        def flush(self) -> None:
            ...

        def write(self, s: _AnyStr_contra) -> Any:
            ...


class Formatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        _super_format = super().format
        
        def _format_line(record: logging.LogRecord, msg: str) -> str:
            record = copy(record)
            record.msg = msg
            record.args = ()
            return _super_format(record)

        return '\n'.join((_format_line(record, msg) for msg in record.getMessage().split('\n')))
    
    
class FileHandler(logging.FileHandler):

    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False, utc: bool = False) -> None:
        if utc:
            t = gmtime()
        else:
            t = localtime()
        filename = strftime(filename, t)
        super().__init__(filename=filename, mode=mode, encoding=encoding, delay=delay)


class Filter(logging.Filter):

    def __init__(self, name: str = '', filter = Callable[[logging.LogRecord], bool]) -> None:
        super().__init__(name)
        self._filter = filter

    def filter(self, record) -> bool:
        return not super().filter(record) or self._filter(record)


class _LogWriter:

    def __init__(self, writer: 'Writer[str]', level: int) -> None:
        self._writer = writer
        self._level = level
        self._msgs: OrderedDict[str, str] = OrderedDict()
        self._local = threading.local()
        self._lock = RLock()

    def write(self, msg: str) -> None:
        if not self._is_logging:
            name = 'root'
            cf = inspect.currentframe()
            if cf is not None:
                f = cf.f_back
                if hasattr(f, 'f_globals'):
                    f_globals = f.f_globals
                    if '__name__' in f_globals and isinstance(name, str):
                        name = f_globals['__name__']
            with self._lock:
                *msg_lines, self._msgs[name] = (self._msgs.get(name, '') + msg).rsplit('\n', maxsplit=1)
            for msg_line in msg_lines:
                self._log(name, msg_line)
        else:
            self._writer.write(msg)

    def flush(self) -> None:
        if not self._is_logging:
            with self._lock:
                msgs = copy(self._msgs)
                self._msgs.clear()
            for name, msg in msgs.items():
                if msg != '':
                    self._log(name, msg)
        else:
            self._writer.flush()

    def _log(self, name: str, msg: str) -> None:
        self._local.is_logging = True
        try:
            logging.getLogger(name).log(self._level, msg)
        finally:
            self._local.is_logging = False

    @property
    def _is_logging(self) -> bool:
        return hasattr(self._local, 'is_logging') and self._local.is_logging


@contextmanager
def _stderr_context(err: TextIO) -> Iterator[TextIO]:
    stderr = sys.stderr
    sys.stderr = err
    try:
        yield sys.stderr
    finally:
        sys.stderr = stderr


@contextmanager
def _stdout_context(out: TextIO) -> Iterator[TextIO]:
    stdout = sys.stdout
    sys.stdout = out
    try:
        yield sys.stdout
    finally:
        sys.stdout = stdout


class _OnContextWriter:

    def __init__(self, context: Callable[[], ContextManager['Writer[str]']]) -> None:
        self.__context = context

    def write(self, s: str) -> None:
        with self.__context() as writer:
            sys.stdout.write(s)
            sys.stdout.flush()

    def flush(self) -> None:
        with self.__context() as writer:
            writer.flush()


class _Displayable():

    def __init__(self, s: str) -> None:
        self._s = s

    def __str__(self) -> str:
        return self._s

    def __repr__(self) -> str:
        return self._s


class _DisplayWriter:

    def write(self, s: str) -> None:
        display(_Displayable(s))

    def flush(self) -> None:
        pass


def _getStderr() -> 'Writer[str]':
    if hasattr(builtins, 'get_ipython'):
        return _DisplayWriter()
    else:
        return sys.__stderr__


def _getStdout() -> 'Writer[str]':
    if hasattr(builtins, 'get_ipython'):
        return _DisplayWriter()
    else:
        return sys.__stdout__


def getStderrHandler() -> Handler:
    return StreamHandler(cast(IO[str], _getStderr()))


def getStdoutHandler() -> Handler:
    return StreamHandler(cast(IO[str], _getStdout()))


def captureStderr(*, level: int = logging.WARNING) -> None:
    sys.stderr = cast(TextIO, _LogWriter(_getStderr(), level))


def captureStdout(*, level: int = logging.INFO) -> None:
    sys.stdout = cast(TextIO, _LogWriter(_getStdout(), level))


def jsonConfig(path: str, pwd: str) -> None:
    with open(path) as f:
        config = json.load(f)
    with pushd(pwd):
        logging.config.dictConfig(config)
    captureStderr()
    captureStdout()
