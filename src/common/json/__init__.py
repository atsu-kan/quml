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


from collections import OrderedDict
import json
from typing import Any


class JSONDecoder(json.JSONDecoder):

    def __init__(self, *, object_pairs_hook=OrderedDict, **kw) -> None:
        super().__init__(object_pairs_hook=object_pairs_hook, **kw)


class JSONEncoder(json.JSONEncoder):

    def default(self, o) -> Any:
        if hasattr(o, '__array__'):
            o = o.__array__().tolist()
        return super().default(o)
