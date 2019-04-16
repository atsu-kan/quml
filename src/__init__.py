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
from datetime import datetime
import logging
from logging import FileHandler, Filter, StreamHandler, getLogger
import os
import os.path
import sys

from src.common.logging import Formatter, captureStderr, captureStdout, getStderrHandler, getStdoutHandler
import src.notebook
from src import pylabzmqinterface
from src import pylabzmqmockclient


OUTDIR: str


def _get_outdir(name: str) -> str:
    titlename, _ = os.path.splitext(os.path.basename(name))
    return os.path.abspath(os.path.join(os.getcwd(), titlename, f"{datetime.now():%Y-%m-%d_%H-%M-%S}"))


def _config(name: str) -> None:
    global OUTDIR
    OUTDIR = _get_outdir(name)
    os.makedirs(OUTDIR, exist_ok=True)
    getLogger().setLevel(logging.INFO)
    formatter = Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(message)s')
    stderr_handler = getStderrHandler()
    stderr_handler.addFilter(lambda record: record.levelno >= logging.WARNING)
    stdout_handler = getStdoutHandler()
    stdout_handler.addFilter(
        lambda record: (
            record.levelno < logging.WARNING
            and not Filter(pylabzmqinterface.__name__).filter(record)
            and not Filter(pylabzmqmockclient.__name__).filter(record)
        )
    )
    file_handler = FileHandler(os.path.join(OUTDIR, 'debug.log'))
    for handler in [stderr_handler, stdout_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        getLogger().addHandler(handler)
    captureStderr()
    captureStdout()

if hasattr(builtins, 'get_ipython'):
    from IPython.display import display, Javascript
    display(Javascript(f'''IPython.notebook.kernel.execute("import sys; sys.modules['{__name__}']._config('" + IPython.notebook.notebook_path + "')")'''))
else:
    _config(sys.argv[0])
