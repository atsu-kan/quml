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


import os.path
from subprocess import Popen
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats

from src.pylabzmqmockclient import run


def experiment(param: pd.Series) -> pd.Series:
    r = np.linalg.norm(param.values)
    return pd.Series([scipy.stats.norm().pdf(r) + 0.0 * scipy.stats.norm().rvs()], index=result_header)


with Popen(['python', 'PyLabZMQInterface.py', 'tcp://127.0.0.1:5555']):

    np.random.seed(1)

    # サーバーへ送る実験パラメータのヘッダー
    param_header = [
        'Labview Param T',
        'Labview Param 1',
        'Labview Param 2',
        'Labview Param 3',
    ]

    # サーバーへ送る実験結果のヘッダー
    result_header = [
        'Labview Result'
    ]

    # クライアントを実行
    run('tcp://127.0.0.1:5555', param_header, result_header, experiment, 2)
