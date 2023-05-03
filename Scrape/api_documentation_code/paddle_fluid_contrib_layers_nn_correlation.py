import paddle
paddle.enable_static()
import paddle.fluid as fluid

x_shape = [3,4]
x_type = 'int32'

x1 = fluid.layers.data(name='x1',
                   shape=[3,4],   # shape = x_shape
                   dtype='int32', # dtype = x_type
                   append_batch_size=False)
x2 = fluid.layers.data(name='x2',
                    shape=[3,4],   # shape = x_shape
                    dtype='int32', # dtype = x_type
                    append_batch_size=False)


out = fluid.contrib.correlation(
                x1,
                x2,
                pad_size=4,
                kernel_size=1,
                max_displacement=4,
                stride1=1,
                stride2=1)