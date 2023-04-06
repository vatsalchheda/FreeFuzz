import paddle.fluid as fluid
x = fluid.layers.fill_constant(shape=[4, 3], value=0.9, dtype='float32')
# x = [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]
y = fluid.layers.fill_constant(
    shape=[4, 1], value=1, dtype='int64')
# y = [[1], [1], [1], [1]]
out = fluid.layers.hsigmoid(input=x, label=y, num_classes=2, param_attr=fluid.initializer.Constant(
    value=0.05), bias_attr=fluid.initializer.Constant(value=.0))
# out = [[0.62792355], [0.62792355], [0.62792355], [0.62792355]]