import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy as np

inp_np = np.ones([5, 200, 100, 100]).astype('float32')
with fluid.dygraph.guard():
    inp_np = to_variable(inp_np)
    prelu0 = fluid.PRelu(
       mode='all',
       param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
    dy_rlt0 = prelu0(inp_np)
    prelu1 = fluid.PRelu(
       mode='channel',
       channel=200,
       param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
    dy_rlt1 = prelu1(inp_np)
    prelu2 = fluid.PRelu(
       mode='element',
       input_shape=inp_np.shape,
       param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
    dy_rlt2 = prelu2(inp_np)