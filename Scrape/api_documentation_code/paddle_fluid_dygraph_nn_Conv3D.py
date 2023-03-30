import paddle.fluid as fluid
import numpy

with fluid.dygraph.guard():
    data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
    conv3d = fluid.dygraph.nn.Conv3D(
          num_channels=3, num_filters=2, filter_size=3, act="relu")
    ret = conv3d(fluid.dygraph.base.to_variable(data))