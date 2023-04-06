import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import numpy

x = numpy.random.random((3, 32, 32)).astype('float32')
with fluid.dygraph.guard():
    x = to_variable(x)
    layerNorm = fluid.LayerNorm([32, 32])
    ret = layerNorm(x)