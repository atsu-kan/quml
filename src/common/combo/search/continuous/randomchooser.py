from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, Tuple, TypeVar

import combo
from combo import variable as Variable
import numpy as np
from scipy.stats import rv_continuous

from .action import Action
from .chooser import Chooser
from .types import GetTest, GetX


if TYPE_CHECKING:
    from typing import Protocol


    class Model(Protocol):

        inf: 'Inf'


    class Inf(Protocol):

        exact: 'Exact'


    class Exact(Protocol):
        ...


class RandomChooser(Chooser):

    def __init__(self, num_data: int, batch_size: int, rv: rv_continuous) -> None:
        self._num_data = num_data
        self._batch_size = batch_size
        self._rv = rv

    def choose_random_actions(self, N: int) -> Action:
        X = self._rv.rvs(N)
        test_X = (X - self._rv.mean()) / self._rv.std()
        return Action(X=X, test=Variable(X=test_X))

    def choose_bayes_action(self, get_test: GetTest) -> Action:

        def one_run(n: int) -> Action:
            actions = self.choose_random_actions(n)
            actions.test = get_test(actions.test.X)
            return actions.get_subset(np.argmax(actions.test.t))

        actions = reduce(lambda a, b: a.add(b), (one_run(n) for n in range(0, self._num_data, self._batch_size)))
        return actions.get_subset(np.argmax(actions.test.t))
