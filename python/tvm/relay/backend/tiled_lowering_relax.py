import tvm
from tvm import relax
# from tvm.driver.build_module import schedule_to_module



def lower_tiled(node : relax.Function, tile_dims : tuple[int,int]) -> tvm.tir.PrimFunc:
    assert len(tile_dims) == 2
    assert tile_dims[0] > 0
    assert tile_dims[1] > 0
    assert isinstance(node, relax.Function)

    # # TODO Adapt to different data formats. Right now assuming H and W are last two dims???
    
    relax.call_tir


    # cached_func = tvm._ffi.get_global_func("relay.backend.LowerToTE")(node)
    # inputs = list(cached_func.inputs)
    # assert len(cached_func.outputs) == 1, "Only single-output nodes supported (Not sure if there is a legitamite reason to tile a multi-output node in this way)"
    # output = cached_func.outputs[0]
    
    # s : tvm.tir.Schedule = tvm.te.create_schedule(output.op)
    # xo, yo, xi, yi = s[output].tile(output.op.axis[-2], output.op.axis[-1], x_factor=tile_dims[0], y_factor=tile_dims[1])
    # import pdb; pdb.set_trace()
    # s.get_loops(s.get_block("c_buffer"))[0]
    # # s.annotate(xo, )
    # # fused = s[output].fuse(xi,yi)


    # lowered_func = tvm.lower(s, inputs + [output])
    # # lowered_func = schedule_to_module(s, list(cached_func.inputs) + list(cached_func.outputs), "main")["main"]
    # import pdb; pdb.set_trace()
    # return lowered_func





# Define relay graph

bb = relax.BlockBuilder()
data = relax.Var("data", relax.TensorStructInfo((1,3,224,224), "int8"))
w0 = relax.Var("w0", relax.TensorStructInfo((8,3,3,3), "int8"))
with bb.dataflow(): 
    conv_output = bb.emit(relax.op.nn.conv2d(data, w0))
    final_output = bb.emit(relax.op.nn.relu(conv_output))

bb.emit_func_output(final_output)
mod = bb.finalize()



# Perform tiled lowering
lowered_func = lower_tiled(mod, (16,16))





