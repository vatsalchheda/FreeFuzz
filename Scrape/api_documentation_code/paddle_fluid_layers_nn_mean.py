import paddle
import paddle.fluid as fluid
paddle.enable_static()

input = fluid.layers.data(
    name='data', shape=[2, 3], dtype='float32')
mean = paddle.mean(input)