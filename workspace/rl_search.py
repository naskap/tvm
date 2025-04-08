# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Evolutionary Search Strategy"""
from tvm._ffi import register_object

from tvm.meta_schedule import _ffi_api
from tvm.meta_schedule.search_strategy.search_strategy import PySearchStrategy, MeasureCandidate
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from tvm.tir.schedule import Schedule
from tvm.meta_schedule.utils import derived_object


@derived_object
class RLSearch(PySearchStrategy):

    def __init__(self):
        pass

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        design_spaces : List[Schedule]
            The design spaces for pre-tuning.
        """
        import pdb; pdb.set_trace()
        raise NotImplementedError

    def post_tuning(self):
        pass

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        raise NotImplementedError

    def notify_runner_results(self, *args, **kwargs):
        pass

    def clone(self):
        return RLSearch()
