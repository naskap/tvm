
import tvm
from tvm import meta_schedule as ms

from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
    get_rules
)
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group
from typing import Dict, Optional, Tuple, Literal, Union, List


def print_sketches_for_workload(mod):
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=tvm.target.Target("cuda --arch=sm_70 --max_threads_per_block=1024"),
        types=None,
        sch_rules = get_rules(kind="cuda-tensorcore", types=ms.schedule_rule.MultiLevelTilingTensorCore) +
                    get_rules(kind="cuda", types=(ms.schedule_rule.MultiLevelTiling, ms.schedule_rule.ParallelizeVectorizeUnroll, ms.schedule_rule.AutoBind)) 
    )
    
    print_sketches(actual)