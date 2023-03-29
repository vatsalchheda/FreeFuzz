import paddle.fluid as fluid
import numpy

with fluid.dygraph.guard():
    data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
    conv3dTranspose = fluid.dygraph.nn.Conv3DTranspose(
           num_channels=3,
           num_filters=12,
           filter_size=12,
           use_cudnn=False)
    ret = conv3dTranspose(fluid.dygraph.base.to_variable(data))