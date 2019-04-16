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


from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy, deepcopy
from enum import IntEnum
from itertools import count, islice, repeat, tee
import os
from os import PathLike
import os.path
import sys
from typing import Any, Callable, Dict, Generator, IO, Iterable, Iterator, List, MutableMapping, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.externals import joblib

from src import OUTDIR
from src.common.combo.policy import Policy
from src.common.combo.predictor import Predictor
from src.common.genertools import Generand, from_generator, call
from src.common.itertools import CopiableIterator


class Exp:

    def __init__(self, search_num: int, probe_num: int, learner_param: pd.Series) -> None:
        self.search_num = search_num
        self.probe_num = probe_num
        self.duplicate_num = 0
        self.learner_param = learner_param
        self.sequencer_param: pd.Series
        self.sequencer_result: pd.Series
        self.learner_result: pd.Series


class LearnerBase(ABC):

    def __init__(self, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed
        self.policy = Policy()
        self.combo_param_header = pd.Index([*self.get_combo_param_header()])
        self.combo_result_header = pd.Index([*self.get_combo_result_header()])
        self.labview_result_header = pd.Index([*self.get_labview_result_header()])
        self.labview_param_header = pd.Index([*self.get_labview_param_header()])
        self.num_duplicates = self.get_num_duplicates()

    def random_search(self, search_num: int, num_probes: int) -> Iterator[Generator[pd.Series, pd.Series, None]]:
        yield from self.__search(search_num, num_probes, random_search)

    def bayes_search(self, search_num: int, num_probes: int, num_candidates: int, score: 'str', interval: int, num_rand_basis: int) -> Iterator[Generator[pd.Series, pd.Series, None]]:
        yield from self.__search(search_num, num_probes, lambda policy: bayes_search(policy, num_candidates, score, interval, num_rand_basis))

    def __search(self, search_num: int, num_probes: int, get_search: Callable[[Policy], Iterator[Callable[[Callable[[int], pd.DataFrame]], Generator[pd.Series, pd.Series, None]]]]) -> Iterator[Generator[pd.Series, pd.Series, None]]:
        for probe_num, limit, search in zip(self.__update_probe_num(num_probes), self.__learner_param_limits, get_search(self.policy)):
            yield from (
                from_generator(search(lambda size: self.__get_candicate_params(limit, size)))
                    .map(lambda learner_param: self.__init_exp(search_num, probe_num, learner_param))
                    .map(lambda exp: self.__save_learner_history(exp, 'history.learner.tsv'))
                    .map(lambda exp: self.__transform_from_to_sequencer(exp))
                    .flat_map(lambda exp: self.__duplicate_from_to_sequencer(exp))
                    .map(lambda exp: self.__save_sequencer_history(exp, 'history.sequencer.tsv'))
                    .map(lambda exp: self.__send_to_sequencer(exp))
            )

    @property
    def __learner_param_limits(self) -> Iterator[Tuple[pd.Series, pd.Series]]:
        for learner_param_limit in islice(self.get_combo_param_limits(), self.policy.training_len, None):
            yield learner_param_limit
        yield from repeat(learner_param_limit)

    def __update_probe_num(self, num_probes) -> Iterator[int]:
        initial_training_len = self.policy.training_len
        for probe_num in count(1 + initial_training_len):
            if self.policy.training_len < initial_training_len + num_probes:
                yield probe_num
            else:
                break

    def __get_candicate_params(self, limit: Tuple[pd.Series, pd.Series], size: int) -> pd.DataFrame:
        low, high = limit
        low = low[self.combo_param_header]
        high = high[self.combo_param_header]
        return pd.DataFrame(np.random.uniform(low.values, high.values, (size, *self.combo_param_header.shape)), columns=self.combo_param_header)

    def __init_exp(self, search_num: int, probe_num: int, learner_param: pd.Series) -> Generator[Exp, Exp, pd.Series]:
        exp = (yield Exp(search_num, probe_num, learner_param))
        return exp.learner_result

    def __transform_from_to_sequencer(self, exp: Exp) -> Generator[Exp, Exp, Exp]:
        exp = copy(exp)
        exp.sequencer_param = self.map_param_from_combo_to_labview(exp.learner_param)[self.labview_param_header]
        exp = (yield exp)
        exp = copy(exp)
        exp.learner_result = self.map_result_from_labview_to_combo(exp.sequencer_result)[self.combo_result_header]
        return exp

    def __duplicate_from_to_sequencer(self, exp: Exp) -> Generator[Iterable[Exp], Iterable[Exp], Exp]:
        def create_exp_generator() -> Iterator[Exp]:
            for i, sequencer_param in enumerate(self.duplicate_param_to_sequencer(exp.sequencer_param)):
                duplicated_exp = copy(exp)
                duplicated_exp.sequencer_param = sequencer_param
                duplicated_exp.duplicate_num = i + 1
                yield duplicated_exp
        exp = copy(exp)
        exp.sequencer_result = self.duplicate_result_from_sequencer([*map(lambda exp: exp.sequencer_result, (yield create_exp_generator()))])
        return exp

    def duplicate_param_to_sequencer(self, sequencer_param: pd.Series) -> Iterable[pd.Series]:
        yield from repeat(sequencer_param, self.num_duplicates)

    def duplicate_result_from_sequencer(self, sequencer_results: Iterable[pd.Series]) -> pd.Series:
        return pd.DataFrame([*sequencer_results]).mean()

    def __save_learner_history(self, exp: Exp, outname: str) -> Generator[Exp, Exp, Exp]:
        exp = (yield exp)
        record = pd.Series(
            [
                self.__get_scan_num(exp),
                self.seed,
                exp.search_num,
                exp.probe_num,
                *exp.learner_param,
                *exp.learner_result
            ],
            index=[
                'scanNum',
                'seed',
                'search_num',
                'probe_num',
                *exp.learner_param.index,
                *exp.learner_result.index
            ],
            dtype=object
        )
        self.__save_history(record, outname)
        return exp

    def __save_sequencer_history(self, exp: Exp, outname: str) -> Generator[Exp, Exp, Exp]:
        exp = (yield exp)
        record = pd.Series(
            [
                self.__get_scan_num(exp),
                self.seed,
                exp.search_num,
                exp.probe_num,
                exp.duplicate_num,
                *exp.sequencer_param,
                *exp.sequencer_result
            ],
            index=[
                'scanNum',
                'seed',
                'search_num',
                'probe_num',
                'duplicate_num',
                *exp.sequencer_param.index,
                *exp.sequencer_result.index
            ],
            dtype=object
        )
        self.__save_history(record, outname)
        return exp

    def __save_history(self, record: pd.Series, outname: str) -> None:
        table = pd.DataFrame([record], dtype=object)
        outpath = os.path.join(OUTDIR, outname)
        if os.path.exists(outpath):
            table.to_csv(outpath, sep='\t', header=False, index=False, mode='a')
        else:
            table.to_csv(outpath, sep='\t', header=True, index=False, mode='w')

    def __send_to_sequencer(self, exp: Exp) -> Generator[pd.Series, pd.Series, Exp]:
        param = pd.Series(
            [
                self.__get_scan_num(exp),
                *exp.sequencer_param
            ],
            index = [
                'scanNum',
                *exp.sequencer_param.index
            ],
            dtype=np.float64
        )
        exp.sequencer_result = (yield param)[self.labview_result_header]
        return exp

    def __get_scan_num(self, exp: Exp) -> int:
        scan_num = self.seed
        scan_num = scan_num * 10 + exp.search_num
        scan_num = scan_num * 1000 + exp.probe_num
        scan_num = scan_num * 10 + exp.duplicate_num
        return scan_num

    @abstractmethod
    def get_combo_param_header(self) -> Iterable[str]:
        ...

    @abstractmethod
    def get_labview_param_header(self) -> Iterable[str]:
        ...

    @abstractmethod
    def get_combo_result_header(self) -> Iterable[str]:
        ...

    @abstractmethod
    def get_labview_result_header(self) -> Iterable[str]:
        ...

    @abstractmethod
    def get_num_duplicates(self) -> int:
        ...

    @abstractmethod
    def get_combo_param_limits(self) -> Iterator[Tuple[pd.Series, pd.Series]]:
        ...

    @abstractmethod
    def map_param_from_combo_to_labview(self, combo_param: pd.Series) -> pd.Series:
        ...

    @abstractmethod
    def map_result_from_labview_to_combo(self, labview_result: pd.Series) -> pd.Series:
        ...


def random_search(policy: Policy) -> Iterator[Callable[[Callable[[int], pd.DataFrame]], Generator[pd.Series, pd.Series, None]]]:
    def probe(get_candidate_params: Callable[[int], pd.DataFrame]) -> Generator[pd.Series, pd.Series, None]:
        X = get_candidate_params(1)
        best_X = X.iloc[0]
        t = (yield best_X)
        policy.write(np.array([best_X.values]), np.array([t.item()]))
    while True:
        yield probe


def bayes_search(policy: Policy, num_candidates: int, score: 'str', interval: int, num_rand_basis: int) -> Iterator[Callable[[Callable[[int], pd.DataFrame]], Generator[pd.Series, pd.Series, None]]]:

    def probe(predictor: Predictor, get_candidate: Callable[[int], pd.DataFrame]) -> Generator[pd.Series, pd.Series, None]:
        get_score = predictor.get_score(score)
        X = get_candidate(num_candidates)
        test = get_score(X.values)
        action = np.argmax(test.t, axis=0)
        best_X = X.iloc[action, :]
        t = (yield best_X)
        policy.write(np.array([best_X.values]), np.array([t.item()]))
        predictor.write(test.get_subset([action]), np.array([t.item()]))

    while True:
        predictor = policy.learn(num_rand_basis=num_rand_basis)
        for _ in range(interval):
            yield lambda get_candidate: probe(predictor, get_candidate)
