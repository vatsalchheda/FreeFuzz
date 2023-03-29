import paddle.fluid as fluid
tmp_tensor = fluid.layers.create_tensor(dtype='float32')
fluid.layers.load(tmp_tensor, "./tmp_tensor.bin")