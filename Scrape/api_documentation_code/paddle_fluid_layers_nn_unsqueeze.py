import paddle.fluid as fluid
x = fluid.layers.data(name='x', shape=[5, 10])
y = fluid.layers.unsqueeze(input=x, axes=[1])