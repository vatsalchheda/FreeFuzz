import paddle
paddle.enable_static()
import paddle.fluid as fluid

label = fluid.layers.data(name='label', shape=[1], dtype='int64')
limit = fluid.layers.fill_constant(shape=[1], value=1, dtype='int64')
out = fluid.layers.not_equal(x=label, y=limit)