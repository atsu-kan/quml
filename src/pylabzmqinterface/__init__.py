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


from typing import Any, Generator, Iterator

import pandas as pd

from src.pylabzmqinterface.adapter import Adapter
from src.pylabzmqinterface.connection import Connection
from src.pylabzmqinterface.pylabinterface import PyLabInterface
from src.pylabzmqinterface.session import Session


def run(binder: Any, on_connection: Iterator[Iterator[Generator[pd.Series, pd.Series, None]]]) -> None:
    PyLabInterface.run(binder, lambda param_header, result_header, initial_param: Adapter(Connection(on_connection, lambda on_session: Session(on_session, param_header, result_header, initial_param))))
