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
# pylint: disable=missing-docstring
import argparse
import logging
from typing import Optional

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.support import describe
from tvm.testing.utils import strtobool
from tvm.meta_schedule.testing.local_rpc import LocalRPC
import print_schedule_space
from tvm.meta_schedule.testing import te_workload
from tvm import te
from rl_search import RLSearch, kernel_gen_add_schedule
from tvm.relax.frontend.torch import from_fx


import os
import numpy as np
import torch
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from torchvision.models.convnext import ConvNeXt_Tiny_Weights, convnext_tiny
import onnx
import tempfile
from tvm.relax.frontend.onnx import from_onnx
import torchvision.models as models


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        required=False,
        help="example: True / False",
        default=True,
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target("nvidia/nvidia-v100", host="llvm")

    # parsed.target = tvm.target.intel_graphics()
    # parsed.rpc_config = ms.runner.RPCConfig(
    #     tracker_host=parsed.rpc_host,
    #     tracker_port=parsed.rpc_port,
    #     tracker_key=parsed.rpc_key,
    #     session_timeout_sec=60,
    # )
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()



def main():
    describe()

    with ms.Profiler() as profiler:
        with LocalRPC() as rpc:
            rpc_runner = ms.runner.RPCRunner(
                rpc_config=ms.runner.RPCConfig(
                    tracker_host=rpc.tracker_host,
                    tracker_port=rpc.tracker_port,
                    tracker_key=rpc.tracker_key,
                ),
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=ARGS.number,
                    repeat=ARGS.repeat,
                    min_repeat_ms=ARGS.min_repeat_ms,
                    enable_cpu_cache_flush=ARGS.cpu_flush,
                ),
                alloc_repeat=3
            )



            # # Export VGG11 to ONNX
            # convnext = models.convnext.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT).eval()
            # dummy_input = torch.randn(1, 3, 224, 224)

            # # Import the model into TVM Relax using from_fx
            # traced = torch.fx.symbolic_trace(convnext)
            # mod = from_fx(traced, [((1,3,224,224),"float16")])

            # mod = relax.get_pipeline("default_build")(mod)

            workload =  te.create_prim_func(
                            te_workload.conv2d_nchw_bias_bn_relu(
                                n=12,
                                h=128,
                                w=136,
                                ci=4,
                                co=8,
                                kh=3,
                                kw=3,
                                stride=2,
                                padding=0,
                                dilation=0,
                                in_dtype="float16",
                            )
                        )
            
            target = ARGS.target

            print_schedule_space.print_sketches_for_workload(workload)

            feature_config = {} # Shared arguments for PerStoreFeatuer and RLModel

            from rl_model import TVMEnv
            env = TVMEnv()
            from tvm.meta_schedule.builder import Builder
            from tvm.meta_schedule.utils import cpu_count
            cpus = cpu_count(logical=False)
            builder = Builder.create(builder, max_workers=cpus)
            rpc_runner

            db : Optional[tir.Schedule] = ms.tir_integration.tune_tir(
                mod=workload,
                target=ARGS.target,
                work_dir=ARGS.work_dir,
                max_trials_global=ARGS.num_trials,
                num_trials_per_iter=64,
                runner=rpc_runner,
                cost_model=ms.cost_model.XGBModel(  # type: ignore
                    extractor=ms.feature_extractor.PerStoreFeature(**feature_config),
                    adaptive_training=ARGS.adaptive_training,
                ),
                strategy=RLSearch(),
            )


            # Copied from tune_tir -- build TuneContext at a higher scope
            # 
            if isinstance(mod, tir.PrimFunc):
                mod = _normalize_mod(mod)

            named_tasks: List[Tuple[str, tir.PrimFunc]] = []
            for gv, func in mod.functions_items():  # pylint: disable=invalid-name
                if isinstance(func, tir.PrimFunc):
                    named_tasks.append((gv.name_hint, func))
            named_tasks.sort(key=lambda x: x[0])

            task_names = [x for x, _ in named_tasks]
            tasks: List[TuneContext] = []
            for task_name, task_func, logger, rand_state in zip(
                task_names,
                [x for _, x in named_tasks],
                get_loggers_from_work_dir(work_dir, task_names),
                fork_seed(seed, n=len(named_tasks)),
            ):
                if special_space and task_name in special_space:
                    task_space = special_space[task_name]
                else:
                    task_space = space
                if task_space is None:
                    continue
                tasks.append(
                    TuneContext(
                        mod=task_func,
                        target=target,
                        space_generator=task_space,
                        search_strategy=strategy,
                        task_name=task_name,
                        rand_state=rand_state,
                        num_threads=num_tuning_cores,
                        logger=logger,
                    ).clone()
                )
            ms.tir_integration.tune_tasks(
            tasks=tasks,
            task_weights=[1.0],
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            builder=builder,
            runner=runner,
            database=database,
            cost_model=cost_model,
            measure_callbacks=measure_callbacks,
            task_scheduler=task_scheduler,
            module_equality=module_equality,
            post_optimization=post_optimization,
        )


            sch = ms.tir_integration.compile_tir(db, workload, target)

    print("Tuning Time:")
    print(profiler.table())

    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    main()
