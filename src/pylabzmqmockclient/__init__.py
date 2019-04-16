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


from typing import Any, Callable, Iterable

import pandas as pd

from src.pylabzmqmockclient.pylabclient import PyLabClient
from src.pylabzmqmockclient.experiment import Experiment


def run(binder: Any, param_header: Iterable[str], result_header: Iterable[str], simulate: Callable[[pd.Series], pd.Series], delay_size: int) -> None:
    _param_header = pd.Index([*param_header])
    _result_header = pd.Index([*result_header])
    PyLabClient.run(binder, _param_header.values, _result_header.values, Experiment(_param_header, _result_header, simulate, delay_size))
