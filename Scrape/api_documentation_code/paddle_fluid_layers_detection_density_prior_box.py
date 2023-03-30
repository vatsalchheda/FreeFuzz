#declarative mode

import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

input = fluid.data(name="input", shape=[None,3,6,9])
image = fluid.data(name="image", shape=[None,3,9,12])
box, var = fluid.layers.density_prior_box(
     input=input,
     image=image,
     densities=[4, 2, 1],
     fixed_sizes=[32.0, 64.0, 128.0],
     fixed_ratios=[1.],
     clip=True,
     flatten_to_2d=True)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# prepare a batch of data
input_data = np.random.rand(1,3,6,9).astype("float32")
image_data = np.random.rand(1,3,9,12).astype("float32")

box_out, var_out = exe.run(
    fluid.default_main_program(),
    feed={"input":input_data,
          "image":image_data},
    fetch_list=[box,var],
    return_numpy=True)

# print(box_out.shape)
# (1134, 4)
# print(var_out.shape)
# (1134, 4)


#imperative mode
import paddle.fluid.dygraph as dg

with dg.guard(place) as g:
    input = dg.to_variable(input_data)
    image = dg.to_variable(image_data)
    box, var = fluid.layers.density_prior_box(
        input=input,
        image=image,
        densities=[4, 2, 1],
        fixed_sizes=[32.0, 64.0, 128.0],
        fixed_ratios=[1.],
        clip=True)

    # print(box.shape)
    # [6L, 9L, 21L, 4L]
    # print(var.shape)
    # [6L, 9L, 21L, 4L]