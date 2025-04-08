import tvm
import tvm._ffi

attributes_schedule : list[dict] = [] # Each index corresponds to a schedule

@tvm._ffi.register_func("kernel_gen_add_schedule")
def kernel_gen_add_schedule(idx, attributes : dict):
    if(idx < len(attributes_schedule) and idx >=0):
        attributes_schedule[idx].update(dict(attributes))
    else:
        assert (idx == len(attributes_schedule)), "idx is {} but len(attributes_schedule) is {}".format(idx, len(attributes_schedule))
        attributes_schedule.append(dict(attributes))

    print(attributes_schedule)




    