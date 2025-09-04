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
# from utils import kernel_gen_add_schedule
# from rl_search import RLSearch
from tvm.target import detect_target

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
        "--alloc-repeat",
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

    args.add_argument(
        "--quick-path",
        action = 'store_true'
    )


    parsed = args.parse_args()
    
    
    if(tvm.cuda().exist):
        parsed.target = detect_target.detect_target_from_device("cuda")
    elif(tvm.opencl().exist):
        parsed.target = detect_target.detect_target_from_device("opencl")
    else:
        parsed.target = detect_target.detect_target_from_device("cpu")

    # parsed.target = tvm.target.Target("nvidia/nvidia-v100", host="llvm")
    # parsed.target = tvm.target.Target("nvidia/t1000")
    # parsed.target = tvm.target.intel_graphics(model="coffeelake_h_gt2")
    parsed.target = tvm.target.Target("llvm -mtriple=x86_64-- -mcpu=core-avx2 -num-cores 12") # Intel laptop processor

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
    import faulthandler
    faulthandler.enable()

    describe()
    with ms.Profiler() as profiler:
        with LocalRPC() as rpc:
            rpc_runner = ms.runner.RPCRunner(
                rpc_config=ms.runner.RPCConfig(
                    tracker_host=rpc.tracker_host,
                    tracker_port=rpc.tracker_port,
                    tracker_key=rpc.tracker_key,
                    session_timeout_sec=60*30
                ),
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=ARGS.number,
                    repeat=ARGS.repeat,
                    min_repeat_ms=ARGS.min_repeat_ms,
                    enable_cpu_cache_flush=ARGS.cpu_flush,
                ),
                alloc_repeat=ARGS.alloc_repeat,
                f_run_evaluator = ms.runner.rpc_runner.f_power_evaluator
            )

            if(ARGS.quick_path):
                workload =  te.create_prim_func(
                                te_workload.conv2d_nchw_bias_bn_relu(
                                    n=1,
                                    h=4,
                                    w=6,
                                    ci=4,
                                    co=8,
                                    kh=3,
                                    kw=3,
                                    stride=1,
                                    padding=1,
                                    dilation=1,
                                    in_dtype="float32",
                                    out_dtype="float32",
                                )
                            )
                strategy = "replay-trace"
            else:
                workload =  te.create_prim_func(
                        te_workload.conv2d_nchw_bias_bn_relu(
                                    n=12,
                                    h=128,
                                    w=136,
                                    ci=3,
                                    co=24,
                                    kh=3,
                                    kw=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    in_dtype="float32",
                                    out_dtype="float32",
                                )
                            )
                strategy = "evolutionary"


            
            # print_schedule_space.print_sketches_for_workload(workload)

            db : Optional[tir.Schedule] = ms.tir_integration.tune_tir(
                mod=ms.tir_integration._normalize_mod(workload),
                target=ARGS.target,
                work_dir=ARGS.work_dir,
                max_trials_global=ARGS.num_trials,
                num_trials_per_iter=64,
                runner=rpc_runner,
                strategy=strategy,
                cost_model=ms.cost_model.XGBModel(  # type: ignore
                    extractor=ms.feature_extractor.PerStoreFeature(),
                    adaptive_training=ARGS.adaptive_training,
                )
            )
            ms.runner.rpc_runner.plot_normalized_results()
            sch = ms.tir_integration.compile_tir(db, workload, ARGS.target)

    print("Tuning Time:")
    print(profiler.table())

    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    main()
