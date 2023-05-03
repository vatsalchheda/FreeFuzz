#declarative mode
import paddle.fluid as fluid
import numpy as np
import paddle

paddle.enable_static()


input = fluid.data(name="input", shape=[None,3,6,9]) 
image = fluid.data(name="image", shape=[None,3,9,12]) 
box, var = fluid.layers.prior_box( input=input, image=image, min_sizes=[100.], clip=True, flip=True)

place = fluid.CPUPlace() 
exe = fluid.Executor(place) 
exe.run(fluid.default_startup_program())

box_out, var_out = exe.run(fluid.default_main_program(), 
                           feed={"input":input,"image":image}, fetch_list=[box,var], return_numpy=True)

print(box_out.shape) # (6, 9, 1, 4) # print(var_out.shape) # (6, 9, 1, 4)

import paddle.fluid.dygraph as dg

with dg.guard(place) as g:
  input = dg.to_variable(input) 
  image = dg.to_variable(image) 
  box, var = fluid.layers.prior_box(
    input=input, image=image, min_sizes=[100.], clip=True, flip=True)