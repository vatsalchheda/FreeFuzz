import paddle.fluid as fluid
x = fluid.data(name='x', shape=[None, 1], dtype='float32')
fc = fluid.layers.fc(input=x, size=10,
    param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))