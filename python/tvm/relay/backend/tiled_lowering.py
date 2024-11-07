import tvm
from tvm import relay
from tvm import tir
# from tvm.driver.build_module import schedule_to_module

# Bring all computations into an outer loop of dimension corresponding to the size of the output
def schedule_to_single_outer_loop(sch : tir.Schedule, output_dims : int):
    
    assert len(sch.get_loops(sch.get_output_blocks("root")[0])) >= output_dims, "{} < {}. ".format(len(sch.get_loops(sch.get_output_blocks("root")[0])), output_dims) + \
                                    "Output block must have at least as many loops as the number of output dimensions specified"

    def recursive_helper(block_to_inline : tir.Block):
        producers = sch.get_producers(block_to_inline)

        for i in range(output_dims):
            sch.compute_at(block_to_inline, sch.get_loops(sch.get_output_blocks("root")[0])[i])
        
        if len(producers) == 0:
            return

        for producer in producers:
            recursive_helper(producer)

    for output_producer_block in sch.get_producers(sch.get_output_blocks("root")[0]):
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

def lower_tiled(node : relay.Function, tile_dims : tuple[int]) -> tir.PrimFunc:
    assert len(tile_dims) == len(node.ret_type.shape)
    assert all([tile_dim > 0 for tile_dim in tile_dims])
    assert isinstance(node, relay.Function)
    assert not isinstance(node.ret_type, tvm.ir.container.Array), "Only single-output nodes supported (Not sure if there is a legitamite reason to tile a multi-output node in this way)"
   
    # Lower Relay function to TensorIR
    cached_func = tvm._ffi.get_global_func("relay.backend.LowerToTE")(node)
    inputs = list(cached_func.inputs)
    output : tvm.te.Tensor = cached_func.outputs[0]
    sch = tir.Schedule(tvm.te.create_prim_func(inputs + [output]))
    out_dims = len(output.op.axis)

    # Initial Transformations
    schedule_to_single_outer_loop(sch, out_dims)
    tile_outer_loop(sch, tile_dims)

    # Inject cache reads and writes (and associated software pipelining annotations)
    in_itervar_deps = get_input_itervar_dependencies(sch)
    inject_cache_reads(sch, in_itervar_deps, out_dims)
    inject_cache_write(sch, out_dims)

    # Inject software pipelining
    mod = tir.transform.InjectSoftwarePipeline()(sch.mod)

    return mod["main"]


# Define relay graph
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
lowered_func = lower_tiled(mod[gv1], (1, 2, 16, 16))
