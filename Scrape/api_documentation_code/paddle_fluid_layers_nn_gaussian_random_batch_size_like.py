import paddle
import paddle.fluid as fluid
paddle.enable_static()

input = fluid.data(name="input", shape=[13, 11], dtype='float32')

out = fluid.layers.gaussian_random_batch_size_like(
    input, shape=[-1, 11], mean=1.0, std=2.0)