import paddle.fluid as fluid
import paddle
paddle.enable_static()
data = fluid.data(name='data', shape=[None, 3, 32, 32],
                         dtype='float32')
output = fluid.layers.im2sequence(
    input=data, stride=[1, 1], filter_size=[2, 2])