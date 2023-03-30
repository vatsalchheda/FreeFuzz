#declarative mode
import paddle
import paddle.fluid as fluid
import numpy as np
paddle.enable_static()
input = fluid.data(name="input", shape=[None,3,6,10])

#1
output = fluid.layers.image_resize(input=input,out_shape=[12,12])

#2
#x = np.array([2]).astype("int32")
#dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
#fluid.layers.assign(input=x, output=dim1)
#output = fluid.layers.image_resize(input=input,out_shape=[12,dim1])

#3
#x = np.array([3,12]).astype("int32")
#shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
#fluid.layers.assign(input=x, output=shape_tensor)
#output = fluid.layers.image_resize(input=input,out_shape=shape_tensor)

#4
#x = np.array([0.5]).astype("float32")
#scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
#fluid.layers.assign(x,scale_tensor)
#output = fluid.layers.image_resize(input=input,scale=scale_tensor)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

input_data = np.random.rand(2,3,6,10).astype("float32")

output_data = exe.run(fluid.default_main_program(),
    feed={"input":input_data},
    fetch_list=[output],
    return_numpy=True)

print(output_data[0].shape)

#1
# (2, 3, 12, 12)
#2
# (2, 3, 12, 2)
#3
# (2, 3, 3, 12)
#4
# (2, 3, 3, 5)

#imperative mode
import paddle.fluid.dygraph as dg

with dg.guard(place) as g:
    input = dg.to_variable(input_data)
    output = fluid.layers.image_resize(input=input, out_shape=[12,12])
    print(output.shape)

    # [2L, 3L, 12L, 12L]