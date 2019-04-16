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


import numpy as np


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-8, atol: float = 1e-8, equal_nan: bool = True) -> np.ndarray:
    return (
        a.shape == b.shape
        and np.all(
            [
                np.isclose(a, b, rtol=rtol, atol=0, equal_nan=equal_nan),
                np.isclose(b, a, rtol=rtol, atol=0, equal_nan=equal_nan),
                np.isclose(a, b, rtol=0, atol=atol, equal_nan=equal_nan)
            ]
        )
    )

def isclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-12, atol: float = 1e-12, equal_nan: bool = True) -> np.ndarray:
    return np.all(
        [
            np.isclose(a, b, rtol=rtol, atol=0, equal_nan=equal_nan),
            np.isclose(b, a, rtol=rtol, atol=0, equal_nan=equal_nan),
            np.isclose(a, b, rtol=0, atol=atol, equal_nan=equal_nan)
        ]
        , axis=0
    )
