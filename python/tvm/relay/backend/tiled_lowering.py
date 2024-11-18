import tvm
from tvm import relay
from tvm import tir
import torch
from torch.nn.modules.rnn import LSTM


# Bring all computations into an outer loop of dimension corresponding to the size of the output
def schedule_to_outer_loop(sch : tir.Schedule, output_dims : int):
    
    assert len(sch.get_loops(sch.get_output_blocks("root")[0])) >= output_dims, "{} < {}. ".format(len(sch.get_loops(sch.get_output_blocks("root")[0])), output_dims) + \
                                    "Output block must have at least as many loops as the number of output dimensions specified"

    inlined_blocks = set()

    def recursive_helper(block_to_inline : tir.Block):

        if(sch.get(block_to_inline) in inlined_blocks):
            return

        for i in range(output_dims):
            sch.compute_at(block_to_inline, sch.get_loops(sch.get_output_blocks("root")[0])[i])

        inlined_blocks.add(sch.get(block_to_inline))

        for producer in reversed(sch.get_producers(block_to_inline)):
            recursive_helper(producer)

    for output_producer_block in reversed(sch.get_producers(sch.get_output_blocks("root")[0])):
        recursive_helper(output_producer_block)

        

# Assuming a schedule is composed of at least len(tile_dims) outer loops, tile those loops according to tile_dims
def tile_outer_loop(sch : tir.Schedule, tile_dims : tuple[int]):
    
    assert len(sch.get_loops(sch.get_output_blocks("root")[0])) >= len(tile_dims), \
            "Schedule has {} outer loops".format(len(sch.get_loops(sch.get_output_blocks("root")[0]))) + \
            "whereas {} tile dims are specified".format(len(tile_dims))

    prev_inners = []
    for i in range(len(tile_dims)):
        outer, inner = sch.split(sch.get_loops(sch.get_output_blocks("root")[0])[2*i], (None, tile_dims[i]))
        if len(prev_inners) > 0:
            sch.reorder(outer, *prev_inners)
        prev_inners.append(inner)


def get_vars_in_stmt(stmt : tir.Stmt) -> list[tir.Var]:

    dependencies = []

    def callback(stmt):
        if isinstance(stmt, tir.Var):
            dependencies.append(stmt)
        return True
    
    tir.stmt_functor.pre_order_visit(stmt, callback)
    return dependencies

# Return a mapping from each given input buffer to the iteration variables used to index that buffer in the given Primfunc
def get_input_itervars(prim_func : tir.PrimFunc, input_buf_list : tir.Buffer) -> dict[tir.Buffer, list[tir.Var]]:
    input_itervars = {in_data : [] for in_data in input_buf_list}

    def callback(node):
        if isinstance(node, tir.BufferLoad) and (node.buffer in input_buf_list):
            for idx in node.indices:
                input_itervars[node.buffer] += get_vars_in_stmt(idx)
        
        prev = node
        return True

    tir.stmt_functor.pre_order_visit(prim_func.body, callback)

    return input_itervars

# Return a mapping from each iteration variable to its variable dependecies
def get_itervar_dependencies(prim_func : tir.PrimFunc):
    
    itervar_tiling_dependencies = {}
    def callback(stmt):
        if isinstance(stmt, tir.BlockRealize):
            for iter_var, iter_value in zip(stmt.block.iter_vars, stmt.iter_values):
                itervar_tiling_dependencies[iter_var.var] = get_vars_in_stmt(iter_value)
        
        return True

    tir.stmt_functor.pre_order_visit(prim_func.body, callback)

    return itervar_tiling_dependencies

# Return a mapping from each input to the given schedule, to its itervar dependencies 
# e.g. If we have the following outer tiling loops : for ax0_0, ax1_0, ax2_0, ax3_0 in T.grid(1, 6, 14, 14):
#      And our reads from data_in are indexed using ax2_0 and ax3_0
#      Then we might return in_outidx_deps[data_in] = [ax2_0, ax3_0]
# Note: Could maybe simplify this by using pass tir.transform.ConvertBlocksToOpaque? 
def get_input_itervar_dependencies(sch : tir.Schedule) -> dict[tir.Buffer, list[tir.Var]]:
    
    # Collect itervars and itervar_deps
    input_itervars = get_input_itervars(sch.mod["main"], list(sch.mod["main"].buffer_map.values()))
    itervar_deps = get_itervar_dependencies(sch.mod["main"])
    
    # Get mapping from inputs to dependencies
    in_itervar_deps = {}
    for input_var in input_itervars:
        in_itervar_deps[input_var] = []
        for itervar in input_itervars[input_var]:
            in_itervar_deps[input_var] += itervar_deps[itervar]
        
        if len(in_itervar_deps[input_var]) == 0:
            del in_itervar_deps[input_var]

    return in_itervar_deps



def inject_cache_reads(sch : tir.Schedule, in_itervar_deps : dict[tir.Buffer, list[tir.Var]], output_dims : int):
    
    # Define helper vars
    sch_inputs = list(in_itervar_deps.keys()) 
    cache_read_injected = {in_data : False for in_data in sch_inputs}

    # Each loop has a "loop_var" of type tir.Var used for iteration
    # Each loop has a "loop_rv" of type tir.schedule.LoopRV used to refer to loops during scheduling
    outer_loop_rvs = [sch.get_loops(sch.get_output_blocks("root")[0])[i] for i in range(output_dims)] 
    outer_loop_vars = [sch.get(loop_rv).loop_var for loop_rv in outer_loop_rvs]

    def add_pipeline_annotations_for_cache_read(loop_var : tir.schedule.LoopRV):
        
        
        # A cache read needs to be in its own stage
        cur_loop_annotations = sch.get(loop_var).annotations
        if(cur_loop_annotations.get("software_pipeline_stage") is not None):
            new_stage_annotation = [0] + [stage + 1 for stage in cur_loop_annotations.get("software_pipeline_stage")]
            sch.unannotate(loop_var, "software_pipeline_stage")
            new_async_annotation = [0] + [stage + 1 for stage in cur_loop_annotations.get("software_pipeline_async_stages")]
            sch.unannotate(loop_var, "software_pipeline_async_stages")
            new_order_annotation = [0] + [order + 1 for order in cur_loop_annotations.get("software_pipeline_order")]
            sch.unannotate(loop_var, "software_pipeline_order")
        else:
            num_stmts_in_loop    = len(sch.get(loop_var).body) 
            new_stage_annotation = [0] + [1 for item in range(num_stmts_in_loop - 1)] # Put cache read in its own stage
            new_async_annotation = [0] # cache read stage should be asynchronous
            new_order_annotation = list(range(num_stmts_in_loop)) # Compute everything in order
            
        sch.annotate(loop_var, "software_pipeline_stage", new_stage_annotation)
        sch.annotate(loop_var, "software_pipeline_async_stages", new_async_annotation)
        sch.annotate(loop_var, "software_pipeline_order", new_order_annotation)

    def inject_cache_reads_per_block(block : tir.Block):

        for buf_idx, buf_region in enumerate(sch.get(block).reads):


            # Select only reads from input buffers
            buffer = buf_region.buffer
            if(buffer not in sch_inputs):
                continue
            
            assert cache_read_injected[buffer] == False, " Not handled -- two reads from an input buffer"
            
            # Determine where the cache_read should be injected using the dependencies of the itervars of the buffer read
            cur_max_restriction = -1
            for itervar_dep in in_itervar_deps[buffer]:

                if(itervar_dep in outer_loop_vars):
                    cur_max_restriction = max(cur_max_restriction, outer_loop_vars.index(itervar_dep))

            # Inject cache read
            cache_block = sch.cache_read(block, buf_idx, storage_scope='local')
            if(cur_max_restriction != -1):
                sch.compute_at(cache_block, outer_loop_rvs[cur_max_restriction]) # Move cache read to location found above
                add_pipeline_annotations_for_cache_read(outer_loop_rvs[cur_max_restriction]) 

            cache_read_injected[buffer] = True


    def recursive_helper(cur_block : tir.Block):
        producers = sch.get_producers(cur_block)

        inject_cache_reads_per_block(cur_block)

        if len(producers) == 0:
            return

        for producer in producers:
            recursive_helper(producer)

    recursive_helper(sch.get_output_blocks("root")[0])
    assert all([injected for injected in cache_read_injected.values()]), " We missed injecting cache read for at least one buffer {}".format(cache_read_injected)

def inject_cache_write(sch : tir.Schedule, out_dims : int):

    inner_tiling_loop = sch.get_loops(sch.get_output_blocks("root")[0])[out_dims - 1]
    
    # Inject cache write
    cache_block = sch.cache_write(sch.get_output_blocks("root")[0], 0, "local")
    sch.reverse_compute_at(cache_block, inner_tiling_loop)

    # Add annotations
    cur_loop_annotations = sch.get(inner_tiling_loop).annotations
    if(cur_loop_annotations.get("software_pipeline_stage") is not None):
        new_stage_annotation = list(cur_loop_annotations.get("software_pipeline_stage"))
        new_stage = max(new_stage_annotation) + 1
        new_stage_annotation.append(new_stage)
        sch.unannotate(inner_tiling_loop, "software_pipeline_stage")

        new_async_annotation = list(cur_loop_annotations.get("software_pipeline_async_stages"))
        new_async_annotation.append(new_stage)
        sch.unannotate(inner_tiling_loop, "software_pipeline_async_stages")

        new_order_annotation = list(cur_loop_annotations.get("software_pipeline_order"))
        new_order_annotation.append(len(new_order_annotation))
        sch.unannotate(inner_tiling_loop, "software_pipeline_order")

    else:
        num_stmts_in_loop = len(sch.get(inner_tiling_loop).body)
        new_stage_annotation = [0 for item in range(num_stmts_in_loop - 1)] + [1] # Put cache_write in its own stage
        new_async_annotation = [1] # Make cache_write asynchronous
        new_order_annotation = list(range(num_stmts_in_loop)) # Compute all stmts in order
        
    sch.annotate(inner_tiling_loop, "software_pipeline_stage", new_stage_annotation)
    sch.annotate(inner_tiling_loop, "software_pipeline_async_stages", new_async_annotation)
    sch.annotate(inner_tiling_loop, "software_pipeline_order", new_order_annotation)

def reannotate_nested_pipeline_loops(sch : tir.Schedule, outer_loop_rv : tir.schedule.LoopRV, inner_loop_rv : tir.schedule.LoopRV):
    
    outer_loop_stmt = sch.get(outer_loop_rv)
    inner_loop_stmt = sch.get(inner_loop_rv)

    assert isinstance(outer_loop_stmt.body, tir.SeqStmt)
    inner_loop_stmt_idx = list(outer_loop_stmt.body).index(inner_loop_stmt)


    # An n stage pipeline has n-1 prologues, a main body, and n-1 epilogues = 2n-1 total "expanded stages" 
    num_inner_loop_stages = max(inner_loop_stmt.annotations.get("software_pipeline_stage"))+1 # + 1 for zero indexing
    num_expanded_pipeline_stages = int(2*num_inner_loop_stages - 1)

    pipeline_order = list(outer_loop_stmt.annotations.get("software_pipeline_order"))


    def insert_list_into_permutation_list(src : list, dest : list, pos : int):
        for idx in range(len(src)):
            
            # Increment all elements greater than the one we are about to insert
            for i in range(len(dest)):
                if(dest[i] >= src[idx]):
                    dest[i]+=1

            dest.insert(pos + idx, src[idx])


    
    to_insert = [pipeline_order[inner_loop_stmt_idx] + i + 1 for i in range(num_expanded_pipeline_stages-1)]
    insert_list_into_permutation_list(to_insert, pipeline_order, inner_loop_stmt_idx+1)
    
    sch.unannotate(outer_loop_rv, "software_pipeline_order")
    sch.annotate(outer_loop_rv, "software_pipeline_order", pipeline_order)

    pipeline_stage = list(outer_loop_stmt.annotations.get("software_pipeline_stage"))
    
    inner_loop_stage = pipeline_stage[inner_loop_stmt_idx]
    for i in range(num_expanded_pipeline_stages-1):
        pipeline_stage.insert(inner_loop_stmt_idx, inner_loop_stage)

    sch.unannotate(outer_loop_rv, "software_pipeline_stage")
    sch.annotate(outer_loop_rv, "software_pipeline_stage", pipeline_stage)



# InjectSoftwarePipeline pass will transform the inner loops before transforming the outer loops.
#   This will split the inner loops into multiple pipeline stages. If we have nested loops
#   with pipeline annotations then the outer loop's annotations need to be adjusted to account for
#   the new number of stmts in the pipeline
def adjust_for_nested_pipelining_loops(sch : tir.Schedule, out_dims : int):

    tiling_loop_rvs = [sch.get_loops(sch.get_output_blocks("root")[0])[i] for i in range(out_dims)] 

    def get_annotation(loop_rv : tir.schedule.LoopRV, key):
        return sch.get(loop_rv).annotations.get(key)

    # iterate from most inner tiling loop to most outer tiling loop in pairs
    inner_loop_rv = tiling_loop_rvs[-1]
    for outer_loop_rv in reversed(tiling_loop_rvs[0:-1]):
        if((get_annotation(outer_loop_rv, "software_pipeline_order") is not None) and 
           (get_annotation(inner_loop_rv,"software_pipeline_order") is not None)):
            
            reannotate_nested_pipeline_loops(sch, outer_loop_rv, inner_loop_rv)

        inner_loop_rv = outer_loop_rv



# Both 'tile_dims' and 'loop_ordering' are with respect to the output shape
def lower_tiled(node : relay.Function, tile_dims : tuple[int], loop_ordering : tuple[int] = None) -> tir.PrimFunc:
    

    assert not isinstance(node.ret_type, tvm.ir.TupleType), "Only single-output nodes supported as it is ambiguous to tile a multi-output node"
    assert isinstance(node.ret_type, tvm.ir.TensorType)
    
    out_dims = len(node.ret_type.shape)
    assert len(tile_dims) == out_dims
    assert all([tile_dim > 0 for tile_dim in tile_dims])
    assert isinstance(node, relay.Function)
   
    if(loop_ordering is None):
        loop_ordering = tuple(range(out_dims))
    
    assert len(loop_ordering) == out_dims

    # Lower Relay function to TensorIR
    cached_func = tvm._ffi.get_global_func("relay.backend.LowerToTE")(node)
    inputs = list(cached_func.inputs)
    output = cached_func.outputs[0]
    sch = tir.Schedule(tvm.te.create_prim_func(inputs + [output]))

    # Bring all computation into outer loop corresponding to size of output
    schedule_to_outer_loop(sch, out_dims)

    # Reorder outer loops to desired order
    outer_loops = sch.get_loops(sch.get_output_blocks("root")[0])[:out_dims]
    outer_loops_reordered = [outer_loops[i] for i in loop_ordering]
    sch.reorder(*outer_loops_reordered)
    
    # Tile outer loop
    tile_dims_reordered = [tile_dims[i] for i in loop_ordering] 
    tile_outer_loop(sch, tile_dims_reordered)

    # Inject cache reads and writes (and associated software pipelining annotations)
    in_itervar_deps = get_input_itervar_dependencies(sch)
    inject_cache_reads(sch, in_itervar_deps, out_dims)
    inject_cache_write(sch, out_dims)
    adjust_for_nested_pipelining_loops(sch, out_dims)

    # Inject software pipelining
    mod = tir.transform.Simplify()(sch.mod)
    import pdb; pdb.set_trace()
    mod = tir.transform.InjectSoftwarePipeline()(mod)
    
    mod = tvm.lower(mod)
    mod = tvm.tir.transform.LoopPartition()(mod) # Not working
    return mod["main"]


def test_conv2dreluconv2drelu():
    data = relay.var("data", relay.TensorType((1,3,224,224), "int8"))
    w0 = relay.var("w0", relay.TensorType((8,3,3,3), "int8"))
    out1 = relay.nn.conv2d(data, w0)
    out1_relu = relay.nn.relu(out1)

    w1 = relay.var("w1", relay.TensorType((12, 8,3,3), "int8"))
    out2 = relay.nn.conv2d(out1_relu, w1)
    out2_relu = relay.nn.relu(out2)

    # Wrap into a function and IRModule
    func = relay.Function(relay.analysis.free_vars(out2_relu), out2_relu)
    mod = tvm.IRModule()
    gv1 = relay.GlobalVar("main")
    mod[gv1] = func

    # Perform tiled lowering
    mod = relay.transform.InferType()(mod)
    lowered_func = lower_tiled(mod[gv1], (1, 2, 16, 16), (1,2,3,0))


def test_dense_relu_layer_norm():
    
    data = relay.var("data", relay.TensorType((4,256,8,8), "float32"))
    out1 = relay.nn.global_avg_pool2d(data)
    out1_flat = relay.nn.batch_flatten(out1)

    w0 = relay.var("w0", relay.TensorType((10, 256), "float32"))
    out2 = relay.nn.dense(out1_flat, w0)
    out2_smax = relay.nn.softmax(out2)

    # Wrap into a function and IRModule
    func = relay.Function(relay.analysis.free_vars(out2_smax), out2_smax)
    mod = tvm.IRModule()
    gv1 = relay.GlobalVar("main")
    mod[gv1] = func


    # Perform tiled lowering
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.SimplifyExpr()(mod)
    mod = relay.transform.InferType()(mod)
    lowered_func = lower_tiled(mod[gv1], (1, 10), (0, 1))



if __name__ == "__main__":
    test_dense_relu_layer_norm()
    test_conv2dreluconv2drelu()
