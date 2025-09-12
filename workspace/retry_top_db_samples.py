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
import os
from tvm.meta_schedule.builder import Builder, BuilderInput
from tvm.meta_schedule.cost_model import CostModel
from tvm.meta_schedule.database import JSONDatabase, Workload

from tvm.script import tir as T
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def _parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument(
    #     "--num-trials",
    #     type=int,
    #     required=True,
    # )
    args.add_argument(
        "--work-dir",
        type=str,
        default="/home/nathan/sandbox/code/tvm/workspace/cuda_latencytuning",
    )
    args.add_argument(
        "--number",
        type=int,
        default=50000,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=10,
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
        "--cpu-flush",
        action = 'store_true'
    )

    args.add_argument(
        "--quick-path",
        action = 'store_true'
    )
    args.add_argument(
        "--cpu",
        action = 'store_true'
    )


    parsed = args.parse_args()
    
    
    if(tvm.cuda().exist and not parsed.cpu):
        parsed.target = detect_target.detect_target_from_device("cuda")
    elif(tvm.opencl().exist and not parsed.cpu):
        parsed.target = detect_target.detect_target_from_device("opencl")
    else:
        # Detection needed to be modified to set num-cores
        parsed.target = detect_target.detect_target_from_device("cpu")
        # target_attrs = dict(target_tmp.attrs)
        # target_attrs["num-cores"] = multiprocessing.cpu_count()
        # target_attrs["kind"] = target_tmp.kind.name
        # parsed.target = tvm.target.Target(target_attrs)
        # parsed.target = tvm.target.Target("llvm -mtriple=x86_64-- -mcpu=core-avx2 -num-cores 12")

    # parsed.target = tvm.target.Target("nvidia/nvidia-v100", host="llvm")
    # parsed.target = tvm.target.Target("nvidia/t1000")
    # parsed.target = tvm.target.intel_graphics(model="coffeelake_h_gt2")
    # parsed.target = tvm.target.Target("llvm -mtriple=x86_64-- -mcpu=core-avx2 -num-cores 12") # Intel laptop processor

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
        energy_runner = ms.runner.LocalRunner(

            evaluator_config=ms.runner.EvaluatorConfig(
                number=ARGS.number,
                repeat=ARGS.repeat,
                min_repeat_ms=ARGS.min_repeat_ms,
                enable_cpu_cache_flush=ARGS.cpu_flush,
            ),
            alloc_repeat=ARGS.alloc_repeat,
            timeout_sec = 60*10,
            f_run_evaluator = ms.runner.local_runner.f_power_evaluator
        )

        latency_runner = ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(
                number=ARGS.number,
                repeat=ARGS.repeat,
                min_repeat_ms=ARGS.min_repeat_ms,
                enable_cpu_cache_flush=ARGS.cpu_flush,
            ),
            alloc_repeat=ARGS.alloc_repeat,
            timeout_sec = 60*10
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

        
        builder = Builder.create("local", max_workers=os.cpu_count())
        database = JSONDatabase(path_workload=f"{ARGS.work_dir}/database_workload.json", path_tuning_record=f"{ARGS.work_dir}/database_tuning_record.json")
        top_10 = database.get_top_k(Workload(ms.tir_integration._normalize_mod(workload)), 100)
        top_10 = [top_10[49]]

        # Build the top-10 tuning records
        meas_candidates : list[ms.MeasureCandidate] = [rec.as_measure_candidate() for rec in top_10]
        build_inputs = [BuilderInput(candidate.sch.mod, ARGS.target) for candidate in meas_candidates]
        build_results = builder.build(build_inputs)
        assert build_results[0].artifact_path is not None, build_results[0].error_msg
        runner_inputs = [ms.runner.RunnerInput(result.artifact_path, ARGS.target.kind.name, candidate.args_info) for result, candidate in zip(build_results, meas_candidates)]

        # Run on RPC
        # energy_results   = [[float(result) for result in future.result().run_secs] if future.result().run_secs is not None else [None] for future in energy_runner.run(runner_inputs) ]
        latency_results  = [[float(result) for result in future.result().run_secs] if future.result().run_secs is not None else [None] for future in latency_runner.run(runner_inputs)] 
        
        # Probably want to get rid of the None elements
        ms.runner.rpc_runner.plot_normalized_results(energy_results, latency_results)
        import pdb; pdb.set_trace()

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
