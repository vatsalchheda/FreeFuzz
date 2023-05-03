import paddle.fluid as fluid
import numpy as np

i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)           # loop counter

loop_len = fluid.layers.fill_constant(shape=[1],dtype='int64', value=10)    # loop length

cond = fluid.layers.less_than(x=i, y=loop_len)
while_op = fluid.layers.While(cond=cond)
with while_op.block():
    i = fluid.layers.increment(x=i, value=1, in_place=True)
    fluid.layers.less_than(x=i, y=loop_len, cond=cond)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])
print(res) # [array([10])]
