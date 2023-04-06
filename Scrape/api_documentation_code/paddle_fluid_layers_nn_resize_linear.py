#declarative mode
import paddle.fluid as fluid
import numpy as np
input = fluid.data(name="input", shape=[None,3,100])

output = fluid.layers.resize_linear(input=input,out_shape=[50,])

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

input_data = np.random.rand(1,3,100).astype("float32")

output_data = exe.run(fluid.default_main_program(),
    feed={"input":input_data},
    fetch_list=[output],
    return_numpy=True)

print(output_data[0].shape)

# (1, 3, 50)

#imperative mode
import paddle.fluid.dygraph as dg

with dg.guard(place) as g:
    input = dg.to_variable(input_data)
    output = fluid.layers.resize_linear(input=input, out_shape=[50,])
    print(output.shape)

    # [1L, 3L, 50L]