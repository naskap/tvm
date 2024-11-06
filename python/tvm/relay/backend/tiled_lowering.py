import tvm
from tvm import relay
from tvm.driver.build_module import schedule_to_module



def lower_tiled(node : relay.Function, tile_dims : tuple[int,int]) -> tvm.tir.PrimFunc:
    assert len(tile_dims) == len(node.ret_type.shape)
    assert all([tile_dim > 0 for tile_dim in tile_dims])
    assert isinstance(node, relay.Function)

    # TODO Adapt to different data formats. Right now assuming NCHW
    
    cached_func = tvm._ffi.get_global_func("relay.backend.LowerToTE")(node)
    inputs = list(cached_func.inputs)
    assert len(cached_func.outputs) == 1, "Only single-output nodes supported (Not sure if there is a legitamite reason to tile a multi-output node in this way)"
    output = cached_func.outputs[0]


    prim_func = tvm.te.create_prim_func(inputs + [output])
    
    sch = tvm.tir.Schedule(prim_func)

    # Bring all computations into an outer loop of dimension corresponding to the size of the output
    block_to_inline = sch.get_output_blocks("root")[0]
    while len(sch.get_producers(block_to_inline)) > 0:
        block_to_inline = sch.get_producers(block_to_inline)[0]
        for i in range(len(output.op.axis)):
            sch.compute_at(block_to_inline, sch.get_loops(sch.get_output_blocks("root")[0])[i])
        
    
    # Tile
    prev_inners = []
    for i in range(len(tile_dims)):
        outer, inner = sch.split(sch.get_loops(sch.get_output_blocks("root")[0])[2*i], (None, tile_dims[i]))
        if len(prev_inners) > 0:
            sch.reorder(outer, *prev_inners)
        prev_inners.append(inner)

    
    # Blockize tiles-- doesn't work
    # sch.blockize(prev_inners[0])
    
    
    
    # xo, yo, xi, yi = sch[output].tile(output.op.axis[-2], output.op.axis[-1], x_factor=tile_dims[0], y_factor=tile_dims[1])
    
    # outer_loops = sch.get_loops(sch.get_output_blocks("root")[0])[len(output.op.axis)-1:len(output.op.axis)]
    # for outer_loop in outer_loops:
    #     sch.annotate(outer_loop, ann_key="software_pipeline_stage", ann_val=[0, 1])
    #     sch.annotate(outer_loop, ann_key="software_pipeline_order", ann_val=[0, 1])
    
    import pdb; pdb.set_trace()
    mod = tvm.tir.transform.InjectSoftwarePipeline()(sch.mod)
    prim_func = mod["main"]

    
    
    # s.annotate(xo, )


    # Old work
    # s : tvm.tir.Schedule = tvm.te.create_schedule(output.op)
    # xo, yo, xi, yi = s[output].tile(output.op.axis[-2], output.op.axis[-1], x_factor=tile_dims[0], y_factor=tile_dims[1])
        # fused = s[output].fuse(xi,yi)
    lowered_func = tvm.lower(sch, inputs + [output])
    # lowered_func = schedule_to_module(s, list(cached_func.inputs) + list(cached_func.outputs), "main")["main"]
    # import pdb; pdb.set_trace()
    return lowered_func





# Define relay graph
data = relay.var("data", relay.TensorType((1,3,224,224), "int8"))
w0 = relay.var("w0", relay.TensorType((8,3,3,3), "int8"))
out1 = relay.nn.conv2d(data, w0, padding = (1,1))
out1_relu = relay.nn.relu(out1)

w1 = relay.var("w1", relay.TensorType((12, 8,3,3), "int8"))
out2 = relay.nn.conv2d(out1_relu, w1, padding = (1,1))
out2_relu = relay.nn.relu(out2)

# Wrap into a function and IRModule
func = relay.Function(relay.analysis.free_vars(out2_relu), out2_relu)
mod = tvm.IRModule()
gv1 = relay.GlobalVar("main")
mod[gv1] = func

# Perform tiled lowering
mod = relay.transform.InferType()(mod)
lowered_func = lower_tiled(mod[gv1], (1, 2, 16, 16))






