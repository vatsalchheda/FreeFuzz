import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
    data = np.random.random((3, 32, 32, 5)).astype('float32')
    conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
          num_channels=32, num_filters=2, filter_size=3)
    ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))