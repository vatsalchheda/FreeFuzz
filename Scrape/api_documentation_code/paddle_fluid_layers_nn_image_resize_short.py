import paddle.fluid as fluid
input = fluid.data(name="input", shape=[None,3,6,9], dtype="float32")
out = fluid.layers.image_resize_short(input, out_short_len=3)