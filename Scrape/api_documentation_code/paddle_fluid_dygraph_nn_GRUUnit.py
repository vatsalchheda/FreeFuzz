import paddle.fluid as fluid
import paddle.fluid.dygraph.base as base
import numpy

lod = [[2, 4, 3]]
D = 5
T = sum(lod[0])

input = numpy.random.rand(T, 3 * D).astype('float32')
hidden_input = numpy.random.rand(T, D).astype('float32')
with fluid.dygraph.guard():
    x = numpy.random.random((3, 32, 32)).astype('float32')
    gru = fluid.dygraph.GRUUnit(size=D * 3)
    dy_ret = gru(
      base.to_variable(input), base.to_variable(hidden_input))