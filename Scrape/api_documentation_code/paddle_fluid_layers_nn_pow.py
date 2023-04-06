import paddle.fluid as fluid

x = fluid.data(name="x", shape=[32,32], dtype="float32")

# example 1: argument factor is float
y_1 = fluid.layers.pow(x, factor=2.0)
# y_1 is x^{2.0}

# example 2: argument factor is Variable
factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
y_2 = fluid.layers.pow(x, factor=factor_tensor)
# y_2 is x^{3.0}