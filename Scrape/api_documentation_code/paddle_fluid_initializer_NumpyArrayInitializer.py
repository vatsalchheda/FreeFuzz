import paddle.fluid as fluid
import numpy
x = fluid.data(name="x", shape=[2, 1], dtype='float32')
fc = fluid.layers.fc(input=x, size=10,
    param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))