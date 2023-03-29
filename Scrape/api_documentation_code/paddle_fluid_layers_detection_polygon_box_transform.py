import paddle.fluid as fluid
input = fluid.data(name='input', shape=[4, 10, 5, 5], dtype='float32')
out = fluid.layers.polygon_box_transform(input)