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


from copy import deepcopy
import sys
from typing import Generator, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd

from src import OUTDIR
from src.learner import LearnerBase
from src.pylabzmqinterface import run


class Learner(LearnerBase):

    def get_combo_param_header(self) -> Iterable[str]:
        """
        Comboに渡す実験パラメータのヘッダー
        """
        return [
            'Combo Param T',
            'Combo Param 1',
            'Combo Param 2',
            'Combo Param 3'
        ]

    def get_labview_param_header(self) -> Iterable[str]:
        """
        Labviewに渡す実験パラメータのヘッダー
        """
        return [
            'Labview Param T',
            'Labview Param 1',
            'Labview Param 2',
            'Labview Param 3',
        ]

    def get_combo_result_header(self) -> Iterable[str]:
        """
        Comboに渡す実験結果のヘッダー
        """
        return [
            'Combo Result',
        ]

    def get_labview_result_header(self) -> Iterable[str]:
        """
        Labviewから受け取る実験結果のヘッダー
        """
        return [
            'Labview Result'
        ]

    def get_num_duplicates(self) -> int:
        """
        同一の実験パラメータで実験を重複して行う回数
        """
        return 2

    def get_combo_param_limits(self) -> Iterator[Tuple[pd.Series, pd.Series]]:
        """
        Comboへ渡す実験パラメータの探索範囲を探索毎に列挙する
        """

        # 実験パラメータTを探索毎に増やす
        for T in np.linspace(0, 1, 101):

            # yield式にて探索毎の探索範囲を順番に指定する
            # for文を併用して、実験パラメータTを探索毎に増やしている
            yield (
                pd.Series([T, -1, -1, -1], index=self.combo_param_header),
                pd.Series([T, +1, +1, +1], index=self.combo_param_header)
            )

    def map_param_from_combo_to_labview(self, combo_param: pd.Series) -> pd.Series:
        """
        ComboからLabviewへ実験パラメータを変換する
        """
        labview_param = pd.Series()
        labview_param['Labview Param T'] = combo_param['Combo Param T'] * 2
        labview_param['Labview Param 1'] = combo_param['Combo Param 1'] * 2
        labview_param['Labview Param 2'] = combo_param['Combo Param 2'] * 2
        labview_param['Labview Param 3'] = combo_param['Combo Param 3'] * 2
        return labview_param

    def map_result_from_labview_to_combo(self, labview_result: pd.Series) -> pd.Series:
        """
        LabviewからComboへ実験結果を変換する。
        """
        combo_result = pd.Series()
        combo_result['Combo Result'] = labview_result['Labview Result']
        return combo_result


def main() -> Iterator[Iterator[Generator[pd.Series, pd.Series, None]]]:

    # seedを1から10まで
    for seed in range(1, 1 + 10):

        #初期状態を生成
        initial_learner = Learner(seed)

        # ランダム探索を20回行う
        yield initial_learner.random_search(
            search_num=1,
            num_probes=20
        )

        # 初期状態をコピー
        learner = deepcopy(initial_learner)

        # 初期状態の続きからランダム探索を80回行う
        yield learner.random_search(
            search_num=1,
            num_probes=80
        )

        # 初期状態をコピー
        learner = deepcopy(initial_learner)

        # 初期状態の続きからベイズ探索を80回行う
        yield learner.bayes_search(
            search_num=2,
            num_probes=80,
            num_candidates=10000,
            score='TS',
            interval=20,
            num_rand_basis=5000
        )


if len(sys.argv) > 1:
    binder = sys.argv[1]
else:
    binder = 'tcp://172.27.25.73:5555'

run(binder, main())
