# declarative mode
import paddle.fluid as fluid
import numpy as np
input = fluid.data(name="input", shape=[2,9,4,4])
output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

input_data = np.random.rand(2,9,4,4).astype("float32")
output_data = exe.run(fluid.default_main_program(),
    feed={"input":input_data},
    fetch_list=[output],
    return_numpy=True)

# print(output.shape)
# (2L, 1L, 12L, 12L)