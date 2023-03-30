import paddle.fluid as fluid

x1 = fluid.layers.data(name='x1',
                   shape=x_shape,
                   dtype=x_type,
                   append_batch_size=False)
x2 = fluid.layers.data(name='x2',
                    shape=x_shape,
                    dtype=x_type,
                    append_batch_size=False)


out = fluid.contrib.correlation(
                x1,
                x2,
                pad_size=4,
                kernel_size=1,
                max_displacement=4,
                stride1=1,
                stride2=1)